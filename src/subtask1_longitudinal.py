import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr

# ==========================================
# CONFIGURATION
# ==========================================
# Users should update these paths to match their local environment
BASE_DIR = Path("./")  # Root of the project
DATA_DIR = BASE_DIR / "data"
SPLIT_OUTPUT_DIR = BASE_DIR / "splits_subtask1"
PREDICTIONS_DIR = BASE_DIR / "predictions"
MODEL_PATH = "best_model.pt"

# Data Files
RAW_DATA_FILE = "train_subtask1.csv"

# Hyperparameters & Settings
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 8
SEQ_LENGTH = 10
MAX_SEQ_LEN = 256
LSTM_HIDDEN = 128
USER_EMBED_DIM = 32
SEED = 42

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. DATA PREPARATION & SPLITTING
# ==========================================

def create_splits():
    """
    Reads raw data, splits users into Seen/Unseen, and creates Train/Val/Test splits.
    """
    raw_path = DATA_DIR / RAW_DATA_FILE
    
    if not raw_path.exists():
        logger.error(f"Raw data file not found at: {raw_path}")
        return

    SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading raw data for splitting...")
    df = pd.read_csv(raw_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Identify Users
    user_ids = df['user_id'].unique()
    np.random.seed(SEED)
    np.random.shuffle(user_ids)

    # Split Users: 20% Unseen, 80% Seen
    n_unseen = int(len(user_ids) * 0.2)
    unseen_users = user_ids[:n_unseen]
    seen_users = user_ids[n_unseen:]

    logger.info(f"Total Users: {len(user_ids)}")
    logger.info(f"Seen Users: {len(seen_users)} | Unseen Users: {len(unseen_users)}")

    # Create 'Unseen' Test Set
    df_test_unseen = df[df['user_id'].isin(unseen_users)].copy()

    # Process 'Seen' Users (Train -> Val -> Test A)
    train_list, val_list, test_seen_list = [], [], []

    for uid in seen_users:
        # Sort strictly by time
        user_df = df[df['user_id'] == uid].sort_values('timestamp')
        n = len(user_df)

        # 80% Train, 10% Val, 10% Test (Forecasting)
        idx_train = int(n * 0.8)
        idx_val = int(n * 0.9)

        train_list.append(user_df.iloc[:idx_train])
        val_list.append(user_df.iloc[idx_train:idx_val])
        test_seen_list.append(user_df.iloc[idx_val:])

    df_train = pd.concat(train_list)
    df_val = pd.concat(val_list)
    df_test_seen = pd.concat(test_seen_list)

    # Save splits
    df_train.to_csv(SPLIT_OUTPUT_DIR / "train.csv", index=False)
    df_val.to_csv(SPLIT_OUTPUT_DIR / "val.csv", index=False)
    df_test_seen.to_csv(SPLIT_OUTPUT_DIR / "test_seen.csv", index=False)
    df_test_unseen.to_csv(SPLIT_OUTPUT_DIR / "test_unseen.csv", index=False)

    logger.info(f"Splits saved to {SPLIT_OUTPUT_DIR}")

# ==========================================
# 2. DATASET & UTILS
# ==========================================

class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, seq_length=10, is_test=False):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.samples = []
        
        # User Mapping Logic (loaded from global scope or passed in)
        # Note: In a full pipeline, pass user_mapper explicitly. 
        # Here we rely on the generic 'get_user_idx' helper.
        
        for uid in df['user_id'].unique():
            u_df = df[df['user_id'] == uid].sort_values('timestamp').reset_index(drop=True)
            texts = u_df["text_cleaned"].fillna("").tolist()

            # For Test/Eval: Predict for EVERY text using padding
            # For Train: Only full sequences
            iterator = range(len(texts)) if is_test else range(len(texts) - seq_length + 1)

            for i in iterator:
                if is_test:
                    start = max(0, i - seq_length + 1)
                    window_texts = texts[start : i+1]
                    # Left padding if window is short
                    if len(window_texts) < seq_length:
                        window_texts = [""] * (seq_length - len(window_texts)) + window_texts

                    self.samples.append({
                        "texts": window_texts,
                        "v": u_df.iloc[i]["valence"], 
                        "a": u_df.iloc[i]["arousal"],
                        "uid": uid
                    })
                else:
                    if len(texts) < seq_length: continue
                    window = u_df.iloc[i : i+seq_length]
                    self.samples.append({
                        "texts": window["text_cleaned"].fillna("").tolist(),
                        "v": window["valence"].tolist(), 
                        "a": window["arousal"].tolist(),
                        "uid": uid
                    })

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        input_ids, masks = [], []
        
        for text in item["texts"]:
            enc = self.tokenizer(
                text, 
                max_length=MAX_SEQ_LEN, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            input_ids.append(enc["input_ids"].squeeze(0))
            masks.append(enc["attention_mask"].squeeze(0))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(masks),
            "user_id": torch.tensor(get_user_idx(item["uid"]), dtype=torch.long),
            "valence": torch.tensor(item["v"], dtype=torch.float),
            "arousal": torch.tensor(item["a"], dtype=torch.float)
        }

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================

class EmotionModel(nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.user_emb = nn.Embedding(num_users, USER_EMBED_DIM)
        
        # Input to LSTM: BERT hidden (768) + User Embed (32)
        lstm_input_dim = 768 + USER_EMBED_DIM
        
        self.lstm = nn.LSTM(
            lstm_input_dim, 
            LSTM_HIDDEN, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.1
        )
        
        # Regression Heads (BiLSTM output = hidden * 2)
        self.head_v = nn.Sequential(
            nn.Linear(LSTM_HIDDEN * 2, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )
        self.head_a = nn.Sequential(
            nn.Linear(LSTM_HIDDEN * 2, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, mask, uid):
        # input_ids: [batch, seq_len, max_len]
        b, s, m = input_ids.shape
        
        # Flatten for BERT
        bert_out = self.bert(input_ids.view(-1, m), mask.view(-1, m)).last_hidden_state[:, 0, :]
        text_emb = bert_out.view(b, s, -1)
        
        # Expand user embedding to match sequence length
        user_emb = self.user_emb(uid).unsqueeze(1).expand(-1, s, -1)
        
        # Early Fusion: Concatenate Text + User
        lstm_out, _ = self.lstm(torch.cat([text_emb, user_emb], dim=-1))
        
        # Predict
        return self.head_v(lstm_out).squeeze(-1), self.head_a(lstm_out).squeeze(-1)

# ==========================================
# 4. UTILITY FUNCTIONS
# ==========================================

# Global user mapping helpers
USER_TO_IDX = {}

def load_user_mapping():
    """Loads user IDs from the training split to create a consistent mapping."""
    train_file = SPLIT_OUTPUT_DIR / "train.csv"
    if train_file.exists():
        train_df_temp = pd.read_csv(train_file)
        known_ids = train_df_temp['user_id'].unique().tolist()
        # 0 is reserved for Unknown/Unseen
        global USER_TO_IDX
        USER_TO_IDX = {uid: i+1 for i, uid in enumerate(known_ids)}
        logger.info(f"User mapping loaded: {len(known_ids)} known users.")
        return len(known_ids) + 1
    else:
        logger.warning("Train split not found. Cannot build user mapping.")
        return 1

def get_user_idx(real_id):
    """Returns the trained index for a user, or 0 if unseen."""
    return USER_TO_IDX.get(real_id, 0)

def run_evaluation(model, dataset_path, split_name, tokenizer):
    logger.info(f"Evaluating on {split_name}...")

    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    # Load Data with is_test=True (Ensures we predict for EVERY text)
    df = pd.read_csv(dataset_path)
    ds = EmotionDataset(df, tokenizer, seq_length=SEQ_LENGTH, is_test=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds_v, all_preds_a = [], []
    all_true_v, all_true_a = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predicting {split_name}"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            uid = batch["user_id"].to(device)

            # Forward Pass
            out_v, out_a = model(input_ids, mask, uid)

            # We only care about the LAST item in the sequence for inference
            last_v = out_v[:, -1].cpu().numpy()
            last_a = out_a[:, -1].cpu().numpy()

            all_preds_v.extend(last_v)
            all_preds_a.extend(last_a)
            all_true_v.extend(batch["valence"].numpy())
            all_true_a.extend(batch["arousal"].numpy())

    # Metrics
    if len(all_true_v) > 1:
        corr_v, _ = pearsonr(all_true_v, all_preds_v)
        corr_a, _ = pearsonr(all_true_a, all_preds_a)
        
        logger.info(f"--- {split_name} RESULTS ---")
        logger.info(f"Valence Correlation: {corr_v:.4f}")
        logger.info(f"Arousal Correlation: {corr_a:.4f}")
    else:
        logger.warning("Not enough data to calculate correlation.")

    # Save to CSV
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    result_df = pd.DataFrame({
        "true_valence": all_true_v, "pred_valence": all_preds_v,
        "true_arousal": all_true_a, "pred_arousal": all_preds_a
    })
    output_file = PREDICTIONS_DIR / f"pred_{split_name}.csv"
    result_df.to_csv(output_file, index=False)
    logger.info(f"Saved predictions to {output_file}")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # A. Create Splits if they don't exist
    if not (SPLIT_OUTPUT_DIR / "train.csv").exists():
        create_splits()
    else:
        logger.info("Splits already exist. Skipping split generation.")

    # B. Initialize Components
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_users = load_user_mapping()
    
    # C. Load Model
    model = EmotionModel(num_users=num_users).to(device)
    
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading weights from {MODEL_PATH}...")
        # Note: weights_only=False due to legacy pickle warnings in some versions
        # Ideally, switch to saving/loading state_dict with weights_only=True in training
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logger.info("Model weights loaded successfully.")
        
        # D. Run Evaluation
        run_evaluation(model, SPLIT_OUTPUT_DIR / "test_seen.csv", "TEST_SEEN", tokenizer)
        run_evaluation(model, SPLIT_OUTPUT_DIR / "test_unseen.csv", "TEST_UNSEEN", tokenizer)
        
    else:
        logger.warning(f"Model file '{MODEL_PATH}' not found. Skipping evaluation.")
        logger.info("To train the model, implement the training loop using the 'train.csv' split.")
