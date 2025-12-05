import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from scipy.stats import pearsonr
from sklearn.model_selection import GroupShuffleSplit

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path("./")
DATA_DIR = BASE_DIR / "data"
SPLIT_OUTPUT_DIR = BASE_DIR / "splits_subtask2a"
MODEL_SAVE_PATH = "best_model_subtask2a.pt"
STATS_SAVE_PATH = "normalization_stats.pt"
PREDICTIONS_DIR = BASE_DIR / "predictions"

# Raw Data File
RAW_DATA_FILE = "train_subtask2a.csv"

# Model Hyperparameters (V5 Configuration)
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 256
TRAIN_BATCH_SIZE = 4      # Optimized for 8GB VRAM
VALID_BATCH_SIZE = 8
ACCUM_STEPS = 8           # Effective batch size = 32
EPOCHS = 6
LR_BACKBONE = 1e-5        # Slower learning for pre-trained layers
LR_HEAD = 1e-3            # Faster learning for regression head
WEIGHT_DECAY = 0.01
SEED = 42

# Feature Config
NUM_NUMERIC_FEATURES = 2  # Current Valence, Current Arousal
NUM_TARGETS = 2           # Delta Valence, Delta Arousal

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. LOSS FUNCTION (CCC LOSS)
# ==========================================
class CCCLoss(nn.Module):
    """
    Concordance Correlation Coefficient Loss.
    Minimizes (1 - CCC).
    Robust to scale differences and strictly penalizes variance mismatch.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        
        mean_true = torch.mean(y_true)
        mean_pred = torch.mean(y_pred)
        
        var_true = torch.var(y_true, unbiased=False)
        var_pred = torch.var(y_pred, unbiased=False)
        
        cov = torch.mean((y_pred - mean_pred) * (y_true - mean_true))
        
        numerator = 2 * cov
        denominator = var_pred + var_true + (mean_pred - mean_true)**2
        
        ccc = numerator / (denominator + 1e-8)
        return 1 - ccc

# ==========================================
# 2. DATA PREPARATION & SPLITTING
# ==========================================
def create_splits():
    """
    Reads raw data, cleans NaNs, and splits by User ID to prevent leakage.
    Creates: train.csv, val.csv, test.csv
    """
    raw_path = DATA_DIR / RAW_DATA_FILE
    if not raw_path.exists():
        logger.error(f"Raw data file not found at: {raw_path}")
        return

    SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading raw data for splitting...")
    df = pd.read_csv(raw_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort strictly by User then Time
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    
    # Drop rows without targets (usually the last post of a user)
    df_clean = df.dropna(subset=['state_change_valence', 'state_change_arousal']).reset_index(drop=True)
    logger.info(f"Original shape: {df.shape} | Cleaned shape: {df_clean.shape}")

    # 1. Split off Test Set (10% of users)
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    train_val_idx, test_idx = next(splitter_test.split(df_clean, groups=df_clean['user_id']))
    
    train_val_df = df_clean.iloc[train_val_idx].reset_index(drop=True)
    test_df = df_clean.iloc[test_idx].reset_index(drop=True)
    
    # 2. Split Train/Val (10% of remaining users)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.11, random_state=SEED)
    train_idx, val_idx = next(splitter_val.split(train_val_df, groups=train_val_df['user_id']))
    
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # Save splits
    train_df.to_csv(SPLIT_OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_OUTPUT_DIR / "test.csv", index=False)
    
    logger.info(f"Train Users: {train_df['user_id'].nunique()} | Val Users: {val_df['user_id'].nunique()} | Test Users: {test_df['user_id'].nunique()}")
    logger.info(f"Splits saved to {SPLIT_OUTPUT_DIR}")

# ==========================================
# 3. DATASET
# ==========================================
class AffectDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, stats=None):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = df['text'].values
        
        # Raw Numeric Features (Current Valence/Arousal)
        raw_current = df[['valence', 'arousal']].values.astype(np.float32)
        
        # Normalization Logic
        if stats is None:
            self.mean = np.mean(raw_current, axis=0)
            self.std = np.std(raw_current, axis=0) + 1e-6
            self.stats = {'mean': self.mean, 'std': self.std}
        else:
            self.mean = stats['mean']
            self.std = stats['std']
            self.stats = stats

        # Normalize Inputs
        self.current_state = (raw_current - self.mean) / self.std
        self.targets = df[['state_change_valence', 'state_change_arousal']].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = str(self.text[index])
        inputs = self.tokenizer(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'numeric_feats': torch.tensor(self.current_state[index], dtype=torch.float),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# ==========================================
# 4. MODEL ARCHITECTURE
# ==========================================
class AffectForecaster(nn.Module):
    def __init__(self, model_name):
        super(AffectForecaster, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        text_hidden = self.config.hidden_size # 768 for Base
        num_hidden = 64 
        
        # 1. Project numeric features so they aren't drowned out
        self.num_projector = nn.Sequential(
            nn.Linear(NUM_NUMERIC_FEATURES, num_hidden),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 2. Fusion Layer
        self.fusion_norm = nn.LayerNorm(text_hidden + num_hidden)
        self.dropout = nn.Dropout(0.3)
        
        # 3. Regressor Head
        self.regressor = nn.Sequential(
            nn.Linear(text_hidden + num_hidden, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_TARGETS)
        )

    def forward(self, ids, mask, numeric_feats):
        # Text Branch
        txt_out = self.backbone(ids, attention_mask=mask)
        txt_embed = txt_out.last_hidden_state[:, 0, :] 
        
        # Numeric Branch
        num_embed = self.num_projector(numeric_feats)   
        
        # Concatenate & Normalize
        fused = torch.cat((txt_embed, num_embed), dim=1)
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)
        
        # Predict
        return self.regressor(fused)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train_model():
    train_path = SPLIT_OUTPUT_DIR / "train.csv"
    val_path = SPLIT_OUTPUT_DIR / "val.csv"
    
    if not train_path.exists():
        logger.error("Splits not found. Please run create_splits() first.")
        return

    logger.info("Loading Datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Initialize Datasets & Capture Normalization Stats
    train_ds = AffectDataset(train_df, tokenizer, MAX_LEN)
    val_ds = AffectDataset(val_df, tokenizer, MAX_LEN, stats=train_ds.stats)
    
    # Save Stats for later Inference
    torch.save(train_ds.stats, STATS_SAVE_PATH)
    logger.info(f"Normalization stats saved to {STATS_SAVE_PATH}")
    
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = AffectForecaster(MODEL_NAME).to(device)
    
    # Differential Learning Rates
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LR_BACKBONE},
        {'params': model.regressor.parameters(), 'lr': LR_HEAD},
        {'params': model.num_projector.parameters(), 'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)
    
    criterion = CCCLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    num_training_steps = int(len(train_ds) / TRAIN_BATCH_SIZE / ACCUM_STEPS * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_training_steps*0.1), num_training_steps=num_training_steps)
    
    best_val_corr = -1.0
    
    logger.info("Starting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for step, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            num_feats = data['numeric_feats'].to(device)
            targets = data['targets'].to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(ids, mask, num_feats)
                loss = criterion(outputs, targets)
                loss = loss / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            train_loss += loss.item() * ACCUM_STEPS
            
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for data in val_loader:
                ids = data['ids'].to(device)
                mask = data['mask'].to(device)
                num_feats = data['numeric_feats'].to(device)
                targets = data['targets'].to(device)
                
                outputs = model(ids, mask, num_feats)
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        
        # Calculate Correlation
        try:
            corr_v, _ = pearsonr(val_preds[:,0], val_targets[:,0])
            corr_a, _ = pearsonr(val_preds[:,1], val_targets[:,1])
            avg_corr = (corr_v + corr_a) / 2
        except:
            avg_corr = 0.0
            
        logger.info(f"Ep {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Corr: {avg_corr:.4f} (V:{corr_v:.2f}, A:{corr_a:.2f})")
        
        if avg_corr > best_val_corr:
            best_val_corr = avg_corr
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"--> Best Model Saved! ({best_val_corr:.4f})")

# ==========================================
# 6. INFERENCE
# ==========================================
def run_inference():
    test_path = SPLIT_OUTPUT_DIR / "test.csv"
    
    if not test_path.exists():
        logger.error("Test split not found.")
        return
        
    if not os.path.exists(MODEL_SAVE_PATH):
        logger.error("Model file not found. Train first.")
        return
        
    logger.info("Running Final Inference on Test Set...")
    df = pd.read_csv(test_path)
    
    # Load Stats for Normalization
    stats = torch.load(STATS_SAVE_PATH, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    ds = AffectDataset(df, tokenizer, MAX_LEN, stats=stats)
    loader = DataLoader(ds, batch_size=VALID_BATCH_SIZE, shuffle=False)
    
    model = AffectForecaster(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Predicting"):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            num_feats = data['numeric_feats'].to(device)
            targets = data['targets'].to(device)
            
            outputs = model(ids, mask, num_feats)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Save Predictions
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    df['pred_val_change'] = all_preds[:, 0]
    df['pred_aro_change'] = all_preds[:, 1]
    
    output_path = PREDICTIONS_DIR / "pred_subtask2a.csv"
    df.to_csv(output_path, index=False)
    
    # Final Metrics
    corr_v, _ = pearsonr(all_preds[:,0], all_targets[:,0])
    corr_a, _ = pearsonr(all_preds[:,1], all_targets[:,1])
    
    logger.info(f"--- FINAL TEST RESULTS ---")
    logger.info(f"Valence Correlation: {corr_v:.4f}")
    logger.info(f"Arousal Correlation: {corr_a:.4f}")
    logger.info(f"Predictions saved to {output_path}")

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Create splits if needed
    if not (SPLIT_OUTPUT_DIR / "train.csv").exists():
        create_splits()
    
    # 2. Train if model doesn't exist
    if not os.path.exists(MODEL_SAVE_PATH):
        train_model()
        
    # 3. Always run inference
    run_inference()
