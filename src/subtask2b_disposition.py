import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from scipy.stats import pearsonr
from sklearn.model_selection import GroupShuffleSplit

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path("./")
DATA_DIR = BASE_DIR / "data"
SPLIT_OUTPUT_DIR = BASE_DIR / "splits_subtask2b"
MODEL_SAVE_PATH = "best_model_leviathan.pt"
SCALER_SAVE_PATH = "target_scaler.pt"
PREDICTIONS_DIR = BASE_DIR / "predictions"

# Raw Data File
RAW_DATA_FILE = "train_subtask2b.csv"

# Model Hyperparameters ("Leviathan" Protocol)
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_SEQ_LEN = 512
MAX_ESSAYS = 32           # Deep History (16 Head + 16 Tail)
TRAIN_BATCH_SIZE = 2      # Low batch size for Large model on consumer GPU
GRAD_ACCUM_STEPS = 8      # Effective batch size = 16
EPOCHS = 5
LR_BACKBONE = 5e-6        # Very slow learning for backbone
LR_HEAD = 1e-4            # Faster learning for heads
WEIGHT_DECAY = 0.01
SEED = 42

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. UTILITIES & LOSS
# ==========================================

class TargetScaler:
    """
    Standardizes targets (Z-score normalization) to help model convergence.
    Crucial for predicting small changes (deltas).
    """
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, targets):
        # targets: numpy array (N, 2)
        self.mean = torch.tensor(np.mean(targets, axis=0), dtype=torch.float).to(device)
        self.std = torch.tensor(np.std(targets, axis=0), dtype=torch.float).to(device)
        
    def transform(self, targets):
        return (targets - self.mean) / (self.std + 1e-8)
    
    def inverse_transform(self, targets):
        # Handle both Tensor and Numpy
        if isinstance(targets, torch.Tensor):
            return targets * (self.std + 1e-8) + self.mean
        else:
            mean = self.mean.cpu().numpy()
            std = self.std.cpu().numpy()
            return targets * (std + 1e-8) + mean

class CCCLoss(nn.Module):
    """
    Concordance Correlation Coefficient Loss.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, y_true, y_pred):
        if y_pred.shape[0] < 2: return self.mse(y_true, y_pred)
            
        ccc_scores = []
        for i in range(y_pred.shape[1]):
            pred = y_pred[:, i]
            true = y_true[:, i]
            
            mu_x, mu_y = torch.mean(true), torch.mean(pred)
            var_x, var_y = torch.var(true), torch.var(pred)
            std_x, std_y = torch.std(true), torch.std(pred)
            
            covariance = torch.mean((true - mu_x) * (pred - mu_y))
            rho = covariance / (std_x * std_y + 1e-8)
            
            numerator = 2 * rho * std_x * std_y
            denominator = var_x + var_y + (mu_x - mu_y)**2
            
            ccc = numerator / (denominator + 1e-8)
            ccc_scores.append(ccc)
            
        return 1.0 - torch.stack(ccc_scores).mean()

# ==========================================
# 2. DATA PREPARATION
# ==========================================
def create_splits():
    raw_path = DATA_DIR / RAW_DATA_FILE
    if not raw_path.exists():
        logger.error(f"Raw data not found: {raw_path}")
        return

    SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading raw data...")
    
    # Read CSV
    # Note: 'text', 'valence', 'arousal' columns are stringified lists in CSV
    df = pd.read_csv(raw_path)
    
    # Safe eval to convert string representation of lists to actual lists
    for col in ['text', 'valence', 'arousal']:
        if isinstance(df[col].iloc[0], str):
            df[col] = df[col].apply(ast.literal_eval)

    # Split Users (10% Test, 10% Val of remainder)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    train_val_idx, test_idx = next(splitter.split(df, groups=df['user_id']))
    
    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.11, random_state=SEED)
    train_idx, val_idx = next(splitter_val.split(train_val_df, groups=train_val_df['user_id']))
    
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # Save to disk (Lists will be saved as strings again, handled in Dataset class)
    train_df.to_csv(SPLIT_OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_OUTPUT_DIR / "test.csv", index=False)
    
    logger.info(f"Splits created: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")

# ==========================================
# 3. DATASET
# ==========================================
class LongitudinalDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_essays, max_seq_len):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_essays = max_essays
        self.max_seq_len = max_seq_len
        
        # Parse Lists (CSV stores them as strings)
        for col in ['text', 'valence', 'arousal']:
            if isinstance(self.df[col].iloc[0], str):
                self.df[col] = self.df[col].apply(ast.literal_eval)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        essays = row['text']
        valences = row['valence']
        arousals = row['arousal']
        
        # --- HEAD + TAIL SAMPLING ---
        # We take the first K/2 (Head) and last K/2 (Tail) essays
        total_essays = len(essays)
        half_k = self.max_essays // 2
        
        if total_essays <= self.max_essays:
            selected_essays = essays
            selected_val = valences
            selected_aro = arousals
        else:
            selected_essays = essays[:half_k] + essays[-half_k:]
            selected_val = valences[:half_k] + valences[-half_k:]
            selected_aro = arousals[:half_k] + arousals[-half_k:]
            
        # --- NAIVE MATH INJECTION ---
        # Calculate raw statistical change to feed into the network
        n_sel = len(selected_val)
        mid = n_sel // 2
        if n_sel > 1:
            naive_val_change = np.mean(selected_val[mid:]) - np.mean(selected_val[:mid])
            naive_aro_change = np.mean(selected_aro[mid:]) - np.mean(selected_aro[:mid])
        else:
            naive_val_change = 0.0
            naive_aro_change = 0.0
            
        naive_features = torch.tensor([naive_val_change, naive_aro_change], dtype=torch.float)

        # Tokenize Essays
        encoded = self.tokenizer(
            selected_essays,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors='pt'
        )
        
        state_features = torch.tensor(list(zip(selected_val, selected_aro)), dtype=torch.float)
        targets = torch.tensor([row['disposition_change_valence'], row['disposition_change_arousal']], dtype=torch.float)
        
        return {
            'input_ids': encoded['input_ids'],      
            'attention_mask': encoded['attention_mask'],
            'state_features': state_features,
            'naive_features': naive_features, 
            'labels': targets
        }

def collate_fn_hybrid(batch):
    # Dynamic padding for batching sequences of essays
    max_n = max([item['input_ids'].shape[0] for item in batch])
    batch_input_ids, batch_mask, batch_states, batch_naive, batch_labels = [], [], [], [], []
    
    for item in batch:
        n, l = item['input_ids'].shape
        pad_n = max_n - n
        
        if pad_n > 0:
            padded_input = torch.cat([item['input_ids'], torch.zeros((pad_n, l), dtype=torch.long)], dim=0)
            padded_mask = torch.cat([item['attention_mask'], torch.zeros((pad_n, l), dtype=torch.long)], dim=0)
            padded_states = torch.cat([item['state_features'], torch.zeros((pad_n, 2), dtype=torch.float)], dim=0)
        else:
            padded_input, padded_mask, padded_states = item['input_ids'], item['attention_mask'], item['state_features']
            
        batch_input_ids.append(padded_input)
        batch_mask.append(padded_mask)
        batch_states.append(padded_states)
        batch_naive.append(item['naive_features'])
        batch_labels.append(item['labels'])
        
    return {
        'input_ids': torch.stack(batch_input_ids),       
        'attention_mask': torch.stack(batch_mask),       
        'state_features': torch.stack(batch_states),     
        'naive_features': torch.stack(batch_naive),
        'labels': torch.stack(batch_labels)              
    }

# ==========================================
# 4. MODEL ARCHITECTURE (THE LEVIATHAN)
# ==========================================
class HierarchicalBifurcated(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.use_cache = False 
        
        # 1. Shared Backbone
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        # Enable Gradient Checkpointing to fit Large model in VRAM
        self.backbone.gradient_checkpointing_enable()
        
        hidden_dim = self.config.hidden_size 
        
        # 2. Bifurcated Heads (Valence vs Arousal)
        self.val_projector = nn.Linear(1, hidden_dim) 
        self.aro_projector = nn.Linear(1, hidden_dim)
        
        # 3. Regression Heads
        # Input size: (Head_Emb + Tail_Emb + Delta_Emb + Naive_Scalar)
        input_dim = hidden_dim * 3 + 1
        
        self.val_head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 512),
            nn.GELU(), nn.Linear(512, 1) 
        )
        
        self.aro_head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 512),
            nn.GELU(), nn.Linear(512, 1) 
        )
        
    def forward(self, input_ids, attention_mask, state_features, naive_features):
        B, N, L = input_ids.shape
        flat_input = input_ids.view(B*N, L)
        flat_mask = attention_mask.view(B*N, L)
        
        # Backbone Forward
        outputs = self.backbone(flat_input, attention_mask=flat_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :] 
        
        # Split State Features
        flat_states = state_features.view(B*N, 2)
        val_states = flat_states[:, 0].unsqueeze(1) 
        aro_states = flat_states[:, 1].unsqueeze(1) 
        
        # Project & Fuse (Residual connection)
        val_emb = cls_embeddings + F.gelu(self.val_projector(val_states))
        aro_emb = cls_embeddings + F.gelu(self.aro_projector(aro_states))
        
        # Helper: Siamese Pooling (Head vs Tail)
        def get_siamese_features(embeddings):
            seq = embeddings.view(B, N, -1)
            half_n = N // 2
            head = torch.mean(seq[:, :half_n, :], dim=1)
            tail = torch.mean(seq[:, half_n:, :], dim=1)
            delta = tail - head
            return head, tail, delta

        v_head, v_tail, v_delta = get_siamese_features(val_emb)
        a_head, a_tail, a_delta = get_siamese_features(aro_emb)
        
        # Inject Naive Features
        naive_val = naive_features[:, 0].unsqueeze(1) 
        naive_aro = naive_features[:, 1].unsqueeze(1) 
        
        # Final Concatenation
        val_input = torch.cat([v_head, v_tail, v_delta, naive_val], dim=1)
        aro_input = torch.cat([a_head, a_tail, a_delta, naive_aro], dim=1)
        
        return torch.cat([self.val_head(val_input), self.aro_head(aro_input)], dim=1)

# ==========================================
# 5. TRAINING ENGINE
# ==========================================
def train_leviathan():
    train_path = SPLIT_OUTPUT_DIR / "train.csv"
    val_path = SPLIT_OUTPUT_DIR / "val.csv"
    
    if not train_path.exists():
        logger.error("Splits not found. Running create_splits()...")
        create_splits()
        if not train_path.exists(): return

    logger.info("Initializing Tokenizer and Datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_ds = LongitudinalDataset(train_path, tokenizer, MAX_ESSAYS, MAX_SEQ_LEN)
    val_ds = LongitudinalDataset(val_path, tokenizer, MAX_ESSAYS, MAX_SEQ_LEN)
    
    # Fit Target Scaler
    logger.info("Fitting Target Scaler...")
    all_targets = []
    for i in range(len(train_ds)):
        all_targets.append(train_ds.df.iloc[i][['disposition_change_valence', 'disposition_change_arousal']].values.astype(float))
    scaler = TargetScaler()
    scaler.fit(np.vstack(all_targets))
    torch.save(scaler, SCALER_SAVE_PATH)
    
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn_hybrid, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate_fn_hybrid, drop_last=True)
    
    logger.info("Building Leviathan Model...")
    model = HierarchicalBifurcated(MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    
    criterion = CCCLoss()
    grad_scaler = GradScaler()
    best_loss = float('inf')
    
    logger.info("Starting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        loss_epoch = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            states = batch['state_features'].to(device)
            naive = batch['naive_features'].to(device)
            raw_labels = batch['labels'].to(device)
            
            # Scale Targets
            scaled_labels = scaler.transform(raw_labels)
            
            with torch.amp.autocast('cuda'):
                preds = model(input_ids, mask, states, naive)
                
                # Split losses for monitoring
                loss_v = criterion(scaled_labels[:, 0].unsqueeze(1), preds[:, 0].unsqueeze(1))
                loss_a = criterion(scaled_labels[:, 1].unsqueeze(1), preds[:, 1].unsqueeze(1))
                
                loss = (loss_v + loss_a) / GRAD_ACCUM_STEPS
            
            grad_scaler.scale(loss).backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()
                
            loss_epoch += loss.item() * GRAD_ACCUM_STEPS
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                states = batch['state_features'].to(device)
                naive = batch['naive_features'].to(device)
                raw_labels = batch['labels'].to(device)
                
                scaled_labels = scaler.transform(raw_labels)
                
                with torch.amp.autocast('cuda'):
                    preds = model(input_ids, mask, states, naive)
                    l_v = criterion(scaled_labels[:, 0].unsqueeze(1), preds[:, 0].unsqueeze(1))
                    l_a = criterion(scaled_labels[:, 1].unsqueeze(1), preds[:, 1].unsqueeze(1))
                    val_loss += (l_v + l_a).item()
        
        avg_val = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(">>> New Best Model Saved!")

# ==========================================
# 6. INFERENCE
# ==========================================
def run_inference():
    test_path = SPLIT_OUTPUT_DIR / "test.csv"
    if not test_path.exists():
        logger.error("Test split not found.")
        return
        
    if not os.path.exists(MODEL_SAVE_PATH):
        logger.error("Model weights not found. Train first.")
        return

    logger.info("Loading Inference Resources...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load Scaler (Critical for Inverse Transform)
    scaler = torch.load(SCALER_SAVE_PATH)
    # Move scaler stats to CPU for final numpy calculation
    scaler.mean = scaler.mean.cpu()
    scaler.std = scaler.std.cpu()
    
    test_ds = LongitudinalDataset(test_path, tokenizer, MAX_ESSAYS, MAX_SEQ_LEN)
    test_loader = DataLoader(test_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate_fn_hybrid)
    
    model = HierarchicalBifurcated(MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds_scaled = []
    all_targets = []
    
    logger.info(f"Predicting for {len(test_ds)} users...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            states = batch['state_features'].to(device)
            naive = batch['naive_features'].to(device)
            
            with torch.amp.autocast('cuda'):
                preds = model(input_ids, mask, states, naive)
            
            all_preds_scaled.append(preds.float().cpu())
            all_targets.append(batch['labels'])
            
    # Concatenate & Inverse Transform
    y_pred_scaled = torch.cat(all_preds_scaled, dim=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()
    
    # Metrics
    corr_v, _ = pearsonr(y_true[:, 0], y_pred[:, 0])
    corr_a, _ = pearsonr(y_true[:, 1], y_pred[:, 1])
    
    logger.info("--- TEST RESULTS ---")
    logger.info(f"Valence Correlation: {corr_v:.4f}")
    logger.info(f"Arousal Correlation: {corr_a:.4f}")
    
    # Save
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    df_res = pd.DataFrame({
        'true_val': y_true[:, 0], 'pred_val': y_pred[:, 0],
        'true_aro': y_true[:, 1], 'pred_aro': y_pred[:, 1]
    })
    df_res.to_csv(PREDICTIONS_DIR / "pred_subtask2b.csv", index=False)
    logger.info(f"Predictions saved to {PREDICTIONS_DIR / 'pred_subtask2b.csv'}")

if __name__ == "__main__":
    if not (SPLIT_OUTPUT_DIR / "train.csv").exists():
        create_splits()
        
    if not os.path.exists(MODEL_SAVE_PATH):
        train_leviathan()
        
    run_inference()
