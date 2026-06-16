import os
import sys
import json
import random
import time
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix,
                             matthews_corrcoef)
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import List
import warnings
warnings.filterwarnings('ignore')

from fusion_models import MultimodalFusionModel
from motif_grouper import extract_5mer, analyze_motifs_by_positive, group_data_by_motif


# ============================================================
#  Global constants
# ============================================================
NUCLEOTIDE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4}
EIIP_DICT = {'A': 0.1260, 'C': 0.1340, 'G': 0.0806, 'T': 0.1335, 'U': 0.1335, 'N': 0.0}
DPP_DIM = 16
STRUCTURE_TO_IDX = {'.': 0, '(': 1, ')': 2, 'N': 3}


# ============================================================
#  Seed fixing - ensure full reproducibility
# ============================================================
SEED = 42

def fix_all_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def check_gpu():
    if not torch.cuda.is_available():
        print("=" * 70)
        print("  Error: GPU / CUDA not available!")
        print("  This program requires GPU. Please check:")
        print("    1. 1. nvidia-smi works")
        print("    2. 2. CUDA driver installed")
        print("    3. 3. PyTorch CUDA version matches")
        print("=" * 70)
        sys.exit(1)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  PyTorch: {torch.__version__}")


# ============================================================
#  Feature computation utilities
# ============================================================
def get_ncp_features(nuc: str) -> list:
    nuc = nuc.upper()
    features = [0] * 12
    if nuc == 'A':
        features[0], features[4], features[8] = 1, 1, 1
    elif nuc == 'C':
        features[1], features[5], features[9] = 1, 1, 1
    elif nuc == 'G':
        features[2], features[6], features[10] = 1, 1, 1
    elif nuc in ['T', 'U']:
        features[3], features[7], features[11] = 1, 1, 1
    return features


def calculate_physicochemical_features(seq: str) -> np.ndarray:
    seq = seq.upper()
    feature_dim = 1 + 12 + 16
    features = np.zeros((len(seq), feature_dim))
    for i, nuc in enumerate(seq):
        if i >= len(seq):
            break
        features[i, 0] = EIIP_DICT.get(nuc, 0.0)
        ncp_feat = get_ncp_features(nuc)
        features[i, 1:13] = ncp_feat
        prev_nuc = seq[i - 1] if i > 0 else 'N'
        dipeptide_idx = (NUCLEOTIDE_TO_IDX.get(prev_nuc, 4) * 5 +
                         NUCLEOTIDE_TO_IDX.get(nuc, 4)) % DPP_DIM
        features[i, 13 + dipeptide_idx] = 1.0
    return features


def _fold_single(seq: str) -> str:
    import RNA
    seq_clean = seq.upper().replace('T', 'U')
    structure, _ = RNA.fold(seq_clean)
    target_len = len(seq_clean)
    if len(structure) > target_len:
        structure = structure[:target_len]
    elif len(structure) < target_len:
        structure += '.' * (target_len - len(structure))
    return structure


def batch_predict_secondary_structure(seqs: List[str], verbose: bool = True,
                                       num_workers: int = 4) -> List[str]:
    total = len(seqs)
    unique_seqs = list(set(seqs))
    seq_to_struct = {}

    if verbose:
        print(f"  Predicting structure for {len(unique_seqs)} unique sequences "
              f"(total {total}, dedup ratio {len(unique_seqs)/total:.2%})...")

    chunk_size = max(1, len(unique_seqs) // (num_workers * 4))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_fold_single, unique_seqs, chunksize=chunk_size))

    for seq, struct in zip(unique_seqs, results):
        seq_to_struct[seq] = struct

    structures = [seq_to_struct[seq] for seq in seqs]
    if verbose:
        print(f"  Structure prediction done: {total} sequences")
    return structures


def encode_secondary_structure(structure: str, seq_len: int = 201) -> np.ndarray:
    feature_dim = 8
    features = np.zeros((seq_len, feature_dim))
    for i, char in enumerate(structure):
        if i >= seq_len:
            break
        idx = STRUCTURE_TO_IDX.get(char, 3)
        features[i, idx] = 1.0
        left_count = sum(1 for c in structure[max(0, i - 3):i] if c == '(')
        right_count = sum(1 for c in structure[i + 1:min(seq_len, i + 4)] if c == ')')
        features[i, 4] = left_count / 3.0
        features[i, 5] = right_count / 3.0
        features[i, 6] = 1.0 if char == '.' else 0.0
        features[i, 7] = 1.0 if char in ['(', ')'] else 0.0
    return features


def normalize_sequence(seq: str, target_len: int = 201) -> str:
    seq = seq.upper().replace('T', 'U')
    if len(seq) < target_len:
        seq = seq + 'N' * (target_len - len(seq))
    elif len(seq) > target_len:
        mid = len(seq) // 2
        half = target_len // 2
        seq = seq[mid - half: mid + half + 1]
    return seq


# ============================================================
#  MotifClassifier — Motif-specific classifier
# ============================================================
class MotifClassifier(nn.Module):
    def __init__(self, input_dim: int = 1536, hidden_dims: list = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 128]
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


# ============================================================
#  Dataset
# ============================================================
class M6AFusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, seq_len: int = 201,
                 max_len: int = 256):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_len = max_len

        print("Preprocessing data...")
        sequences = []
        labels = []
        for _, row in df.iterrows():
            seq = normalize_sequence(str(row['text']).strip(), seq_len)
            sequences.append(seq)
            labels.append(int(row['label']))

        print(f"  Batch predicting secondary structure ({len(sequences)} seqs)...")
        structures = batch_predict_secondary_structure(sequences, verbose=True, num_workers=4)

        print("  Computing features + tokenization...")
        self.processed_data = []
        for i in range(len(sequences)):
            seq = sequences[i]
            structure = structures[i]
            label = labels[i]

            phy_features = calculate_physicochemical_features(seq).astype(np.float32)
            str_features = encode_secondary_structure(structure, seq_len).astype(np.float32)

            kmer_text = " ".join([seq[j:j + 3] for j in range(len(seq) - 2)])
            encoded = self.tokenizer(
                kmer_text, padding='max_length', truncation=True,
                max_length=self.max_len, return_tensors='pt'
            )

            self.processed_data.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'phy_features': phy_features,
                'str_features': str_features,
                'label': label
            })

            if (i + 1) % 5000 == 0:
                print(f"    Processed {i + 1}/{len(sequences)} samples...")

        print(f"Preprocessing done, total {len(self.processed_data)} samples")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        data = self.processed_data[idx]
        return {
            'input_ids': data['input_ids'],
            'attention_mask': data['attention_mask'],
            'phy_features': torch.tensor(data['phy_features'], dtype=torch.float32),
            'str_features': torch.tensor(data['str_features'], dtype=torch.float32),
            'label': torch.tensor(data['label'], dtype=torch.float32)
        }


class PredictionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, seq_len: int = 201,
                 max_len: int = 256, text_col: str = None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_len = max_len

        if text_col is None:
            text_col = 'text' if 'text' in df.columns else df.columns[0]
        sequences = []
        for _, row in df.iterrows():
            seq = normalize_sequence(str(row[text_col]).strip(), seq_len)
            sequences.append(seq)

        print(f"  Batch predicting secondary structure ({len(sequences)} seqs)...")
        structures = batch_predict_secondary_structure(sequences, verbose=True, num_workers=4)

        print("  Computing features + tokenization...")
        self.processed_data = []
        for i in range(len(sequences)):
            seq = sequences[i]
            structure = structures[i]

            phy_features = calculate_physicochemical_features(seq).astype(np.float32)
            str_features = encode_secondary_structure(structure, seq_len).astype(np.float32)

            kmer_text = " ".join([seq[j:j + 3] for j in range(len(seq) - 2)])
            encoded = self.tokenizer(
                kmer_text, padding='max_length', truncation=True,
                max_length=self.max_len, return_tensors='pt'
            )

            self.processed_data.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'phy_features': phy_features,
                'str_features': str_features,
            })

            if (i + 1) % 5000 == 0:
                print(f"    Processed {i + 1}/{len(sequences)} samples...")

        print(f"Preprocessing done, total {len(self.processed_data)} samples")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        data = self.processed_data[idx]
        return {
            'input_ids': data['input_ids'],
            'attention_mask': data['attention_mask'],
            'phy_features': torch.tensor(data['phy_features'], dtype=torch.float32),
            'str_features': torch.tensor(data['str_features'], dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'phy_features': torch.stack([item['phy_features'] for item in batch]),
        'str_features': torch.stack([item['str_features'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch])
    }


def collate_fn_predict(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'phy_features': torch.stack([item['phy_features'] for item in batch]),
        'str_features': torch.stack([item['str_features'] for item in batch]),
    }


# ============================================================
#  Training utilities
# ============================================================
def train_epoch_fusion(model, dataloader, optimizer, scheduler, scaler, device,
                       alpha_recon, epoch):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_recon_loss = 0.0
    all_preds = []
    all_labels = []

    bce_fn = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            outputs = model(batch)
            cls_loss = bce_fn(outputs['pred_logits'], batch['labels'])
            recon_loss = (mse_fn(outputs['recon_phy'], outputs['phy_feature_map']) +
                          mse_fn(outputs['recon_str'], outputs['str_feature_map']))
            loss = cls_loss + alpha_recon * recon_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_recon_loss += recon_loss.item()

        with torch.no_grad():
            probs = torch.sigmoid(outputs['pred_logits']).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].cpu().numpy())

        if (step + 1) % 50 == 0:
            print(f"  Epoch [{epoch}] Step [{step + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} (Cls: {cls_loss.item():.4f}, "
                  f"Recon: {recon_loss.item():.4f})")

    n = len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5

    return {
        'loss': total_loss / n, 'cls_loss': total_cls_loss / n,
        'recon_loss': total_recon_loss / n, 'accuracy': acc, 'auc': auc
    }


@torch.no_grad()
def evaluate_fusion(model, dataloader, device, alpha_recon):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    bce_fn = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with autocast(device_type='cuda'):
            outputs = model(batch)
            cls_loss = bce_fn(outputs['pred_logits'], batch['labels'])
            recon_loss = (mse_fn(outputs['recon_phy'], outputs['phy_feature_map']) +
                          mse_fn(outputs['recon_str'], outputs['str_feature_map']))
            loss = cls_loss + alpha_recon * recon_loss

        total_loss += loss.item()
        probs = torch.sigmoid(outputs['pred_logits']).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch['labels'].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    preds = (np.array(all_probs) > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average='binary', zero_division=0
    )

    return {
        'loss': avg_loss, 'accuracy': acc, 'auc': auc,
        'precision': precision, 'recall': recall, 'f1': f1
    }


def train_classifier(model, train_loader, val_loader, device, lr, weight_decay,
                     patience, group_name, seed):
    fix_all_seeds(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    bce_fn = nn.BCELoss()

    best_auc = 0.0
    best_state = None
    best_metrics = {}
    patience_counter = 0
    max_epochs = 50

    for epoch in range(1, max_epochs + 1):
        fix_all_seeds(seed + epoch)
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = bce_fn(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend((preds.detach() > 0.5).cpu().numpy().astype(int))
            train_labels.extend(y_batch.cpu().numpy())

        train_loss /= len(train_loader)
        try:
            train_auc = roc_auc_score(train_labels, train_preds)
        except ValueError:
            train_auc = 0.5

        if val_loader is None or len(val_loader.dataset) == 0:
            val_auc = train_auc
            val_loss = train_loss
            val_acc = accuracy_score(train_labels, train_preds)
            _, _, val_f1, _ = precision_recall_fscore_support(
                train_labels, train_preds, average='binary', zero_division=0)
        else:
            model.eval()
            val_loss = 0.0
            val_probs = []
            val_labels = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds = model(X_batch)
                    loss = bce_fn(preds, y_batch)
                    val_loss += loss.item()
                    val_probs.extend(preds.cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())

            val_loss /= len(val_loader)
            val_preds = (np.array(val_probs) > 0.5).astype(int)
            try:
                val_auc = roc_auc_score(val_labels, val_probs)
            except ValueError:
                val_auc = 0.5
            val_acc = accuracy_score(val_labels, val_preds)
            _, _, val_f1, _ = precision_recall_fscore_support(
                val_labels, val_preds, average='binary', zero_division=0
            )

        scheduler.step(val_auc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: Train Loss={train_loss:.4f} AUC={train_auc:.4f} | "
                  f"Val Loss={val_loss:.4f} AUC={val_auc:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {'auc': val_auc, 'acc': val_acc, 'f1': val_f1}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_auc, best_state, best_metrics


def build_dataloader(features: np.ndarray, labels: np.ndarray,
                     indices: list, batch_size: int, shuffle: bool = True):
    if len(indices) == 0:
        return None
    X = torch.tensor(features[indices], dtype=torch.float32)
    y = torch.tensor(labels[indices], dtype=torch.float32)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader


@torch.no_grad()
def extract_features_from_model(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with autocast(device_type='cuda'):
            outputs = model(batch)

        all_features.append(outputs['F_combined'].cpu().numpy())
        if 'labels' in batch and batch['labels'] is not None:
            all_labels.extend(batch['labels'].cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels) if all_labels else None
    return features, labels


# ============================================================
#   — Train all models from scratch
# ============================================================
def main_train(
    data_path: str = "./all_train_samples.tsv",
    model_name: str = "./DNABERT3",
    output_dir: str = ".",
    seed: int = SEED,
    fusion_epochs: int = 20,
    fusion_batch_size: int = 16,
    fusion_lr: float = 5e-4,
    fusion_patience: int = 5,
    classifier_batch_size: int = 64,
    classifier_lr: float = 1e-3,
    classifier_patience: int = 7,
    motif_threshold_ratio: float = 0.015,
):
    """
    Train all models from scratch ( DNABERT3 ):
      Step 1: Train multimodal fusion trunk (CNN + sparse gating + reconstruction decoder)
      Step 2: Extract 1536-dim features using fusion trunk
      Step 3: Group by positive sample motif frequency
      Step 4: Train motif-specific MLP classifier for each group
    Fixed seeds throughout; all models and results saved for full reproducibility.
    """
    t0 = time.time()
    fix_all_seeds(seed)
    check_gpu()

    device = torch.device('cuda')

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    feat_dir = os.path.join(output_dir, "features")
    group_dir = os.path.join(output_dir, "grouped_data")
    clf_dir = os.path.join(output_dir, "classifiers")
    results_dir = os.path.join(output_dir, "results")

    for d in [ckpt_dir, feat_dir, group_dir, clf_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    trunk_path = os.path.join(ckpt_dir, "multimodal_fusion_trunk.pt")
    features_path = os.path.join(feat_dir, "m6a_features.npz")
    routing_map_path = os.path.join(clf_dir, "routing_map.json")
    training_log_path = os.path.join(results_dir, "training_log.json")

    SEQ_LEN = 201
    MAX_LEN = 256
    EMBEDDING_DIM = 768
    PHY_FEATURE_DIM = 29
    STR_FEATURE_DIM = 8
    CNN_CHANNELS_PHY = [128, 256, 512]
    CNN_CHANNELS_STR = [64, 128, 256]
    KERNEL_SIZE = 3
    RECON_PHY_CHANNELS = [256, 512]
    RECON_STR_CHANNELS = [128, 256]
    RECON_TARGET_LENGTH = 25
    CLASSIFIER_HIDDEN_DIM = 256
    DROPOUT = 0.3
    ALPHA_RECON = 0.1

    training_log = {
        'seed': seed, 'device': str(device),
        'data_path': data_path, 'model_name': model_name,
        'steps': {}
    }

    print("=" * 70)
    print("  m6A HMF - Full Training Pipeline")
    print("=" * 70)
    print(f"  Device: {device}, Seed: {seed}")
    print(f"  Output: {output_dir}")

    # ---- Load data ----
    print("\n[Step 0] Loading training data...")
    if data_path.endswith('.tsv'):
        df = pd.read_csv(data_path, sep='\t')
    else:
        df = pd.read_csv(data_path)
    print(f"  Dataset: {len(df)} samples (pos:{(df['label'] == 1).sum()} / neg:{(df['label'] == 0).sum()})")

    # ================================================================
    #  Step 1: Fusion trunk
    # ================================================================
    print(f"\n[Step 1] Training multimodal fusion trunk...")
    t1 = time.time()

    fix_all_seeds(seed)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df['label']
    )
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("  Creating training dataset...")
    train_dataset = M6AFusionDataset(train_df, tokenizer, SEQ_LEN, MAX_LEN)
    print("  Creating validation dataset...")
    val_dataset = M6AFusionDataset(val_df, tokenizer, SEQ_LEN, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=fusion_batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=fusion_batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)

    fix_all_seeds(seed)
    model = MultimodalFusionModel(
        model_name=model_name,
        embedding_dim=EMBEDDING_DIM,
        phy_input_dim=PHY_FEATURE_DIM,
        phy_channels=CNN_CHANNELS_PHY,
        str_input_dim=STR_FEATURE_DIM,
        str_channels=CNN_CHANNELS_STR,
        kernel_size=KERNEL_SIZE,
        recon_phy_channels=RECON_PHY_CHANNELS,
        recon_str_channels=RECON_STR_CHANNELS,
        recon_target_length=RECON_TARGET_LENGTH,
        classifier_hidden=CLASSIFIER_HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)

    for param in model.bert_extractor.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}, Trainable: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=fusion_lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader) * 5, T_mult=2
    )
    scaler = GradScaler('cuda')

    best_auc = 0.0
    patience_counter = 0
    epoch_logs = []

    for epoch in range(1, fusion_epochs + 1):
        print(f"\n  {'='*40}")
        print(f"  Epoch {epoch}/{fusion_epochs}")
        print(f"  {'='*40}")

        train_metrics = train_epoch_fusion(
            model, train_loader, optimizer, scheduler, scaler, device,
            ALPHA_RECON, epoch
        )
        val_metrics = evaluate_fusion(model, val_loader, device, ALPHA_RECON)

        sparsity_phy = model.gate_phy.element_linear.get_sparsity()
        sparsity_str = model.gate_str.element_linear.get_sparsity()

        print(f"  [Epoch {epoch}] Train: Loss={train_metrics['loss']:.4f} "
              f"Cls={train_metrics['cls_loss']:.4f} "
              f"Recon={train_metrics['recon_loss']:.4f} "
              f"AUC={train_metrics['auc']:.4f} Acc={train_metrics['accuracy']:.4f}")
        print(f"  [Epoch {epoch}] Val:   Loss={val_metrics['loss']:.4f} "
              f"AUC={val_metrics['auc']:.4f} Acc={val_metrics['accuracy']:.4f} "
              f"Precision={val_metrics['precision']:.4f} Recall={val_metrics['recall']:.4f} "
              f"F1={val_metrics['f1']:.4f}")
        print(f"  [Epoch {epoch}] Gate sparsity: Phy={sparsity_phy:.4f} Str={sparsity_str:.4f}")

        epoch_logs.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_auc': train_metrics['auc'],
            'val_loss': val_metrics['loss'],
            'val_auc': val_metrics['auc'],
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
        })

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0

            trunk_state = model.get_fusion_trunk_state()
            torch.save({
                'fusion_trunk': trunk_state,
                'epoch': epoch,
                'best_auc': best_auc,
                'config': {
                    'EMBEDDING_DIM': EMBEDDING_DIM,
                    'PHY_FEATURE_DIM': PHY_FEATURE_DIM,
                    'STR_FEATURE_DIM': STR_FEATURE_DIM,
                    'CNN_CHANNELS_PHY': CNN_CHANNELS_PHY,
                    'CNN_CHANNELS_STR': CNN_CHANNELS_STR,
                    'KERNEL_SIZE': KERNEL_SIZE,
                    'RECON_PHY_CHANNELS': RECON_PHY_CHANNELS,
                    'RECON_STR_CHANNELS': RECON_STR_CHANNELS,
                    'RECON_TARGET_LENGTH': RECON_TARGET_LENGTH,
                    'CLASSIFIER_HIDDEN_DIM': CLASSIFIER_HIDDEN_DIM,
                    'DROPOUT': DROPOUT,
                }
            }, trunk_path)
            print(f"  * New best model! AUC: {best_auc:.4f}, trunk saved")
        else:
            patience_counter += 1
            print(f"  Val AUC not improved ({patience_counter}/{fusion_patience})")

        if patience_counter >= fusion_patience:
            print(f"\n  Early stopping! Best AUC: {best_auc:.4f}")
            break

    t1_elapsed = time.time() - t1
    training_log['steps']['step1_fusion_trunk'] = {
        'status': 'trained', 'path': trunk_path,
        'best_auc': float(best_auc), 'elapsed_sec': round(t1_elapsed, 1),
        'epoch_logs': epoch_logs
    }
    print(f"\n  Step 1 done! Best val AUC: {best_auc:.4f}, elapsed: {t1_elapsed:.1f}s")

    del train_dataset, val_dataset, train_loader, val_loader
    torch.cuda.empty_cache()

    # ================================================================
    #  Step 2:  1536 
    # ================================================================
    print(f"\n[Step 2] Extracting 1536-dim features...")
    t2 = time.time()

    ckpt = torch.load(trunk_path, map_location=device, weights_only=False)
    model_config = ckpt['config']

    model = MultimodalFusionModel(
        model_name=model_name,
        embedding_dim=model_config['EMBEDDING_DIM'],
        phy_input_dim=model_config['PHY_FEATURE_DIM'],
        phy_channels=model_config['CNN_CHANNELS_PHY'],
        str_input_dim=model_config['STR_FEATURE_DIM'],
        str_channels=model_config['CNN_CHANNELS_STR'],
        kernel_size=model_config['KERNEL_SIZE'],
        recon_phy_channels=model_config['RECON_PHY_CHANNELS'],
        recon_str_channels=model_config['RECON_STR_CHANNELS'],
        recon_target_length=model_config['RECON_TARGET_LENGTH'],
        classifier_hidden=model_config['CLASSIFIER_HIDDEN_DIM'],
        dropout=model_config['DROPOUT']
    ).to(device)

    for param in model.bert_extractor.parameters():
        param.requires_grad = False
    model.load_fusion_trunk_state(ckpt['fusion_trunk'])
    print(f"  Fusion trunk loaded (epoch={ckpt.get('epoch', '?')}, AUC={ckpt.get('best_auc', '?')})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = M6AFusionDataset(df, tokenizer, SEQ_LEN, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=fusion_batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)

    features, labels = extract_features_from_model(model, dataloader, device)
    print(f"  Feature matrix: {features.shape}")

    motifs = []
    sequences = []
    for _, row in df.iterrows():
        seq = str(row['text']).strip()
        motifs.append(extract_5mer(seq))
        sequences.append(seq)

    np.savez(features_path,
             features=features, labels=labels,
             motifs=np.array(motifs), sequences=np.array(sequences))
    print(f"  Features saved: {features_path}")

    t2_elapsed = time.time() - t2
    training_log['steps']['step2_feature_extraction'] = {
        'status': 'done', 'path': features_path,
        'shape': list(features.shape), 'elapsed_sec': round(t2_elapsed, 1)
    }
    print(f"  Step 2 done! elapsed: {t2_elapsed:.1f}s")

    del dataset, dataloader
    torch.cuda.empty_cache()

    # ================================================================
    #  Step 3: Group by motif
    # ================================================================
    print(f"\n[Step 3] Grouping by positive sample motif frequency (threshold = num_positive * {motif_threshold_ratio})...")
    t3 = time.time()

    df_with_motif, major_motifs, has_others = analyze_motifs_by_positive(
        df, threshold_ratio=motif_threshold_ratio
    )

    group_names = group_data_by_motif(
        df_with_motif, major_motifs, has_others, group_dir
    )

    motif_to_indices = defaultdict(list)
    for idx, row in df_with_motif.iterrows():
        motif = row['motif']
        if motif in major_motifs:
            group_key = motif
        else:
            group_key = 'others'
        motif_to_indices[group_key].append(idx)

    index_map_path = os.path.join(group_dir, 'feature_index_map.json')
    with open(index_map_path, 'w') as f:
        json.dump({k: v for k, v in motif_to_indices.items()}, f, indent=2)

    t3_elapsed = time.time() - t3
    training_log['steps']['step3_motif_grouping'] = {
        'status': 'done', 'num_groups': len(group_names),
        'major_motifs': major_motifs, 'has_others': has_others,
        'elapsed_sec': round(t3_elapsed, 1)
    }
    print(f"  Step 3 done! {len(group_names)} motif groups, elapsed: {t3_elapsed:.1f}s")

    # ================================================================
    #  Step 4: Motif-specific classifier
    # ================================================================
    print(f"\n[Step 4] Motif-specific classifier...")
    t4 = time.time()

    INPUT_DIM = 1536
    HIDDEN_DIMS = [512, 128]
    CLF_DROPOUT = 0.3

    routing_map = {}
    clf_results = {}

    for group_name in group_names:
        print(f"\n  {'-'*50}")
        print(f"  Motif: {group_name}")
        print(f"  {'-'*50}")

        indices = motif_to_indices.get(group_name, [])
        if len(indices) == 0:
            print(f"  Skip: no samples")
            continue

        group_labels = labels[indices]
        pos_count = int((group_labels == 1).sum())
        neg_count = int((group_labels == 0).sum())
        print(f"  Samples: {len(indices)} (pos:{pos_count} / neg:{neg_count})")

        if pos_count < 2 or neg_count < 2:
            print(f"  : pos/neg (>=2)")
            continue

        indices_arr = np.array(indices)
        pos_indices = indices_arr[group_labels == 1]
        neg_indices = indices_arr[group_labels == 0]

        np.random.seed(seed)
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        n_pos_train = max(1, int(len(pos_indices) * 0.8))
        n_neg_train = max(1, int(len(neg_indices) * 0.8))

        train_indices = np.concatenate([
            pos_indices[:n_pos_train], neg_indices[:n_neg_train]
        ]).tolist()
        val_indices = np.concatenate([
            pos_indices[n_pos_train:], neg_indices[n_neg_train:]
        ]).tolist()

        train_loader = build_dataloader(
            features, labels, train_indices, classifier_batch_size, shuffle=True
        )
        val_loader = build_dataloader(
            features, labels, val_indices, classifier_batch_size, shuffle=False
        )

        if train_loader is None:
            continue

        fix_all_seeds(seed)
        clf_model = MotifClassifier(
            input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout=CLF_DROPOUT
        ).to(device)

        best_auc, best_state, best_metrics = train_classifier(
            clf_model, train_loader, val_loader, device,
            classifier_lr, 1e-4, classifier_patience, group_name, seed
        )

        safe_name = group_name.replace('/', '_').replace('\\', '_')
        weight_path = os.path.join(clf_dir, f"classifier_{safe_name}.pt")
        torch.save({
            'model_state_dict': best_state,
            'input_dim': INPUT_DIM,
            'hidden_dims': HIDDEN_DIMS,
            'dropout': CLF_DROPOUT,
            'best_auc': best_auc,
            'best_acc': best_metrics.get('acc', 0.0),
            'best_f1': best_metrics.get('f1', 0.0),
            'group_name': group_name,
            'num_samples': len(indices),
            'pos_count': pos_count,
            'neg_count': neg_count
        }, weight_path)

        routing_map[group_name] = weight_path
        clf_results[group_name] = {
            'auc': float(best_auc),
            'acc': float(best_metrics.get('acc', 0.0)),
            'f1': float(best_metrics.get('f1', 0.0)),
            'n_samples': len(indices),
            'pos': pos_count,
            'neg': neg_count
        }

        print(f"  Best AUC: {best_auc:.4f} Acc: {best_metrics.get('acc', 0.0):.4f} "
              f"F1: {best_metrics.get('f1', 0.0):.4f}")

    with open(routing_map_path, 'w') as f:
        json.dump({
            'routing': routing_map,
            'input_dim': INPUT_DIM,
            'hidden_dims': HIDDEN_DIMS,
            'dropout': CLF_DROPOUT,
            'default_group': 'others' if 'others' in routing_map else None
        }, f, indent=2)

    t4_elapsed = time.time() - t4

    if clf_results:
        aucs = [v['auc'] for v in clf_results.values()]
        accs = [v['acc'] for v in clf_results.values()]
        f1s = [v['f1'] for v in clf_results.values()]
        avg_auc = float(np.mean(aucs))
        avg_acc = float(np.mean(accs))
        avg_f1 = float(np.mean(f1s))
    else:
        avg_auc = avg_acc = avg_f1 = 0.0

    training_log['steps']['step4_motif_classifiers'] = {
        'status': 'done', 'num_classifiers': len(routing_map),
        'avg_auc': avg_auc, 'avg_acc': avg_acc, 'avg_f1': avg_f1,
        'per_group': clf_results,
        'elapsed_sec': round(t4_elapsed, 1)
    }

    with open(training_log_path, 'w') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    if clf_results:
        summary_path = os.path.join(results_dir, "classifier_summary.csv")
        rows = []
        for name, info in sorted(clf_results.items()):
            rows.append({
                'motif': name, 'n_samples': info['n_samples'],
                'pos': info['pos'], 'neg': info['neg'],
                'auc': info['auc'], 'acc': info['acc'], 'f1': info['f1']
            })
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"\n  Classifier summary saved: {summary_path}")

    total_elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  Training complete!")
    print(f"  elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  Fusion trunk: {trunk_path}")
    print(f"  Features file: {features_path}")
    print(f"  Motif groups: {group_dir}")
    print(f"  Classifiers dir: {clf_dir}")
    print(f"  Routing map: {routing_map_path}")
    print(f"  Training log: {training_log_path}")
    if clf_results:
        print(f"\n  Per-group performance summary:")
        print(f"  {'Motif':<12} {'Samples':>8} {'pos/neg':>10} {'AUC':>8} {'ACC':>8} {'F1':>8}")
        print(f"  {'-'*58}")
        for name, info in sorted(clf_results.items()):
            print(f"  {name:<12} {info['n_samples']:>8} {info['pos']:>4}/{info['neg']:<4} "
                  f"{info['auc']:>8.4f} {info['acc']:>8.4f} {info['f1']:>8.4f}")
        print(f"\n  Mean AUC: {avg_auc:.4f}")
        print(f"  Mean ACC: {avg_acc:.4f}")
        print(f"  Mean F1:  {avg_f1:.4f}")
    print("=" * 70)

    return training_log


# ============================================================
#  5-mer extraction (uppercase + U->T)
# ============================================================
def extract_5mer_upper_ut(seq, motif_size=5):
    seq = str(seq).strip().upper().replace('U', 'T')
    length = len(seq)
    mid_idx = length // 2
    if length >= motif_size:
        half = motif_size // 2
        return seq[mid_idx - half: mid_idx + half + 1]
    return seq


# ============================================================
#  Main prediction function — File
# ============================================================
def main_predict(
    test_dir: str = "./test_motif_results",
    model_dir: str = ".",
    dnabert_path: str = "./DNABERT3",
    seed: int = SEED,
    batch_size: int = 16,
    single_input: str = None,
    input_file: str = None,
    input_column: str = None,
):
    """
    Predicting:
      -  input_file:  test_motif_results/  CSV Predicting ( main.py )
      -  input_file + input_column: FilePredicting

    Pipeline:
      1.  + Motif
      2. File:  1536  →  5-mer Motif groups → Predicting
      3. FilePredicting + Evaluation
    """
    fix_all_seeds(seed)
    t0 = time.time()
    check_gpu()

    device = torch.device('cuda')

    ckpt_dir = os.path.join(model_dir, "checkpoints")
    clf_dir = os.path.join(model_dir, "classifiers")
    pred_dir = os.path.join(model_dir, "results")
    os.makedirs(pred_dir, exist_ok=True)

    if input_file is not None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        summary_filename = f"pred_{base_name}_summary.csv"
    else:
        summary_filename = "all_predictions_summary.csv"

    trunk_path = os.path.join(ckpt_dir, "multimodal_fusion_trunk.pt")
    routing_map_path = os.path.join(clf_dir, "routing_map.json")

    print("=" * 70)
    print("  m6A HMF — Prediction pipeline")
    print("=" * 70)
    print(f"  Device: {device}, Seed: {seed}")

    # ---- 1.  ----
    print("\n[Step 1] Loading fusion model...")
    if not os.path.exists(trunk_path):
        print(f"  : Fusion trunk {trunk_path}")
        print(f"  Please run main_train() first")
        return None

    ckpt = torch.load(trunk_path, map_location=device, weights_only=False)
    model_config = ckpt['config']
    print(f"  Training Best AUC: {ckpt.get('best_auc', 'N/A')}")

    model = MultimodalFusionModel(
        model_name=dnabert_path,
        embedding_dim=model_config['EMBEDDING_DIM'],
        phy_input_dim=model_config['PHY_FEATURE_DIM'],
        phy_channels=model_config['CNN_CHANNELS_PHY'],
        str_input_dim=model_config['STR_FEATURE_DIM'],
        str_channels=model_config['CNN_CHANNELS_STR'],
        kernel_size=model_config['KERNEL_SIZE'],
        recon_phy_channels=model_config['RECON_PHY_CHANNELS'],
        recon_str_channels=model_config['RECON_STR_CHANNELS'],
        recon_target_length=model_config['RECON_TARGET_LENGTH'],
        classifier_hidden=model_config['CLASSIFIER_HIDDEN_DIM'],
        dropout=model_config['DROPOUT']
    ).to(device)

    for param in model.bert_extractor.parameters():
        param.requires_grad = False
    model.load_fusion_trunk_state(ckpt['fusion_trunk'])
    print("  Fusion trunk loaded")

    # ---- 2. Motif ----
    print("\n[Step 2] Motif...")
    if not os.path.exists(routing_map_path):
        print(f"  : Routing map {routing_map_path}")
        return None

    with open(routing_map_path, 'r') as f:
        routing_info = json.load(f)

    routing = routing_info['routing']
    hidden_dims = routing_info['hidden_dims']
    dropout_val = routing_info['dropout']
    default_group = routing_info.get('default_group', 'others')

    classifiers = {}
    for motif_name, weight_path in routing.items():
        weight_path_fixed = weight_path.replace('\\', '/')

        if not os.path.exists(weight_path_fixed):
            alt_path = os.path.join(clf_dir, f"classifier_{motif_name}.pt")
            if os.path.exists(alt_path):
                weight_path_fixed = alt_path
            else:
                print(f"  Warning: classifier weights not found, skipping {motif_name}")
                continue

        clf_ckpt = torch.load(weight_path_fixed, map_location=device, weights_only=False)
        clf = MotifClassifier(
            input_dim=routing_info['input_dim'],
            hidden_dims=hidden_dims, dropout=dropout_val
        ).to(device)
        clf.load_state_dict(clf_ckpt['model_state_dict'])
        clf.eval()
        classifiers[motif_name] = clf

    print(f"  Loaded {len(classifiers)} classifiers: {list(classifiers.keys())}")
    major_motifs = set(routing.keys())

    # ---- 3. Determine files to predict ----
    is_custom_input = (input_file is not None)

    if is_custom_input:
        if not os.path.exists(input_file):
            print(f"  Error: input file not found {input_file}")
            return None
        csv_files = [input_file]
        print(f"\n[Step 3] Custom input mode")
        print(f"  File: {input_file}")
        print(f"  Sequence column: {input_column}")
    elif single_input:
        csv_files = [single_input]
        print(f"\n[Step 3] FilePredicting: {single_input}")
    else:
        csv_files = sorted(glob.glob(os.path.join(test_dir, "test_*.csv")))
        if not csv_files:
            print(f"  : {test_dir}  test_*.csv File")
            return None
        print(f"\n[Step 3] Files to predict: {len(csv_files)} ")
        for f in csv_files:
            print(f"    {os.path.basename(f)}")

    tokenizer = AutoTokenizer.from_pretrained(dnabert_path)

    # ---- 4. Per-file prediction ----
    all_file_results = []
    global_true_labels = []
    global_all_probs = []
    global_all_preds = []

    for file_idx, csv_path in enumerate(csv_files):
        file_name = os.path.basename(csv_path)
        print(f"\n{'='*70}")
        print(f"  [{file_idx+1}/{len(csv_files)}] Predicting: {file_name}")
        print(f"{'='*70}")

        if csv_path.endswith('.tsv'):
            df = pd.read_csv(csv_path, sep='\t')
        else:
            df = pd.read_csv(csv_path)

        if is_custom_input and input_column is not None:
            text_col = input_column
        else:
            text_col = 'text' if 'text' in df.columns else df.columns[0]

        if text_col not in df.columns:
            print(f"  :  '{text_col}' File")
            print(f"  Available columns: {list(df.columns)}")
            continue

        has_label = 'label' in df.columns
        print(f"  Samples: {len(df)}, Sequence column: {text_col}, labels: {'' if has_label else ''}")

        # Motif ( + U->T)
        df['motif_5mer'] = df[text_col].apply(
            lambda x: extract_5mer_upper_ut(str(x).strip())
        )
        df['motif_group'] = df['motif_5mer'].apply(
            lambda m: m if m in major_motifs else default_group
        )

        group_counts = df['motif_group'].value_counts()
        print(f"  Motif groups:")
        for g, c in group_counts.items():
            print(f"    {g}: {c}")

        #  1536 
        print(f"  Extracting 1536-dim features...")
        pred_df = df[[text_col]].copy()
        if has_label:
            pred_df['label'] = df['label']
        pred_df = pred_df.rename(columns={text_col: 'text'})

        dataset = PredictionDataset(pred_df, tokenizer, seq_len=201, max_len=256,
                                    text_col='text')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn_predict, num_workers=0,
                                pin_memory=True)

        all_features, feat_labels = extract_features_from_model(model, dataloader, device)
        print(f"  Feature matrix: {all_features.shape}")

        del dataset, dataloader
        torch.cuda.empty_cache()

        # Group by motifPredicting
        print(f"  Group by motifPredicting...")
        all_probs = np.zeros(len(df))
        all_preds = np.zeros(len(df), dtype=int)

        for group_name, clf in classifiers.items():
            mask = df['motif_group'] == group_name
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            X = torch.tensor(all_features[indices], dtype=torch.float32).to(device)
            clf_probs = []
            for i in range(0, len(X), 256):
                batch_x = X[i:i + 256]
                with torch.no_grad():
                    p = clf(batch_x).cpu().numpy()
                clf_probs.extend(p)

            clf_probs = np.array(clf_probs)
            clf_preds = (clf_probs > 0.5).astype(int)

            all_probs[indices] = clf_probs
            all_preds[indices] = clf_preds

            n_pos_pred = int(clf_preds.sum())
            print(f"    [{group_name}] {len(indices)} , PredictingposExample: {n_pos_pred}, "
                  f"Mean: {clf_probs.mean():.4f}")

        # Predicting
        result_df = df.copy()
        result_df['m6A_prob'] = all_probs
        result_df['m6A_pred'] = all_preds
        result_df['motif_group'] = df['motif_group']

        if is_custom_input:
            base_name = os.path.splitext(file_name)[0]
            out_name = f"pred_{base_name}_predictions.csv"
        else:
            out_name = file_name.replace('test_', 'pred_').replace('.csv', '_predictions.csv')
        out_path = os.path.join(pred_dir, out_name)
        result_df.to_csv(out_path, index=False)
        print(f"  Predicting: {out_path}")

        # Evaluation
        file_result = {'file': file_name, 'n_samples': len(df)}
        if has_label:
            true_labels = df['label'].values.astype(int)

            acc = accuracy_score(true_labels, all_preds)
            try:
                auc = roc_auc_score(true_labels, all_probs)
            except ValueError:
                auc = 0.5
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, all_preds, average='binary', zero_division=0
            )
            cm = confusion_matrix(true_labels, all_preds)
            mcc = matthews_corrcoef(true_labels, all_preds)

            print(f"\n  === Evaluation ===")
            print(f"  AUC:       {auc:.4f}")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1:        {f1:.4f}")
            print(f"  MCC:       {mcc:.4f}")
            print(f"  Confusion Matrix:")
            print(f"           Pred_Neg  Pred_Pos")
            print(f"  Actual_Neg  {cm[0,0]:>6}   {cm[0,1]:>6}")
            print(f"  Actual_Pos  {cm[1,0]:>6}   {cm[1,1]:>6}")

            file_result.update({
                'auc': auc, 'acc': acc, 'precision': precision,
                'recall': recall, 'f1': f1, 'mcc': mcc,
                'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])
            })

            global_true_labels.append(true_labels)
            global_all_probs.append(all_probs)
            global_all_preds.append(all_preds)

        all_file_results.append(file_result)

    # ---- 5. Summary ----
    print(f"\n{'='*70}")
    print("  FilePredicting!")
    print(f"{'='*70}")

    if all_file_results:
        summary_path = os.path.join(pred_dir, summary_filename)
        pd.DataFrame(all_file_results).to_csv(summary_path, index=False)
        print(f"  : {summary_path}")

        has_metrics = [r for r in all_file_results if 'auc' in r]
        if has_metrics:
            print(f"\n  === FileEvaluation ===")
            print(f"  {'File':<25} {'N':>6} {'AUC':>7} {'Acc':>7} {'F1':>7}")
            print(f"  {'-'*55}")
            for r in has_metrics:
                print(f"  {r['file']:<25} {r['n_samples']:>6} {r['auc']:>7.4f} "
                      f"{r['acc']:>7.4f} {r['f1']:>7.4f}")

            mean_auc = np.mean([r['auc'] for r in has_metrics])
            mean_acc = np.mean([r['acc'] for r in has_metrics])
            mean_f1 = np.mean([r['f1'] for r in has_metrics])
            print(f"\n  Mean AUC: {mean_auc:.4f}")
            print(f"  Mean ACC: {mean_acc:.4f}")
            print(f"  Mean F1:  {mean_f1:.4f}")

        if global_true_labels:
            g_labels = np.concatenate(global_true_labels)
            g_probs = np.concatenate(global_all_probs)
            g_preds = np.concatenate(global_all_preds)
            g_acc = accuracy_score(g_labels, g_preds)
            try:
                g_auc = roc_auc_score(g_labels, g_probs)
            except ValueError:
                g_auc = 0.5
            g_precision, g_recall, g_f1, _ = precision_recall_fscore_support(
                g_labels, g_preds, average='binary', zero_division=0
            )
            g_cm = confusion_matrix(g_labels, g_preds)
            g_mcc = matthews_corrcoef(g_labels, g_preds)

            print(f"\n  === Evaluation（）===")
            print(f"  Samples:  {len(g_labels)}")
            print(f"  AUC:       {g_auc:.4f}")
            print(f"  Accuracy:  {g_acc:.4f}")
            print(f"  Precision: {g_precision:.4f}")
            print(f"  Recall:    {g_recall:.4f}")
            print(f"  F1:        {g_f1:.4f}")
            print(f"  MCC:       {g_mcc:.4f}")
            print(f"  Confusion Matrix:")
            print(f"           Pred_Neg  Pred_Pos")
            print(f"  Actual_Neg  {g_cm[0,0]:>6}   {g_cm[0,1]:>6}")
            print(f"  Actual_Pos  {g_cm[1,0]:>6}   {g_cm[1,1]:>6}")

            global_result = {
                'file': 'GLOBAL_OVERALL', 'n_samples': len(g_labels),
                'auc': g_auc, 'acc': g_acc, 'precision': g_precision,
                'recall': g_recall, 'f1': g_f1, 'mcc': g_mcc,
                'tn': int(g_cm[0, 0]), 'fp': int(g_cm[0, 1]),
                'fn': int(g_cm[1, 0]), 'tp': int(g_cm[1, 1])
            }
            all_file_results.append(global_result)

            summary_path = os.path.join(pred_dir, summary_filename)
            pd.DataFrame(all_file_results).to_csv(summary_path, index=False)
            print(f"\n  EvaluationFile: {summary_path}")

    total_elapsed = time.time() - t0
    print(f"\n  Predictingelapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print("=" * 70)

    return all_file_results


# ============================================================
#  
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  conda activate m6A")
        print("  python main_general.py train                                          # Train all models from scratch")
        print("  python main_general.py predict                                        # Predict all CSVs in test_motif_results/")
        print("  python main_general.py predict <file_path> <column_name>              # PredictingFile")
        print("  Example: python main_general.py predict HIV_data/xxx.csv context_find")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'train':
        kwargs = {}
        if '--seed' in sys.argv:
            kwargs['seed'] = int(sys.argv[sys.argv.index('--seed') + 1])
        if '--epochs' in sys.argv:
            kwargs['fusion_epochs'] = int(sys.argv[sys.argv.index('--epochs') + 1])
        main_train(**kwargs)

    elif command == 'predict':
        kwargs = {}

        # : python main_general.py predict [file_path] [column_name]
        positional_args = []
        for arg in sys.argv[2:]:
            if arg.startswith('--'):
                break
            positional_args.append(arg)

        if len(positional_args) >= 2:
            kwargs['input_file'] = positional_args[0]
            kwargs['input_column'] = positional_args[1]
        elif len(positional_args) == 1:
            kwargs['single_input'] = positional_args[0]

        if '--test_dir' in sys.argv:
            kwargs['test_dir'] = sys.argv[sys.argv.index('--test_dir') + 1]
        if '--model_dir' in sys.argv:
            kwargs['model_dir'] = sys.argv[sys.argv.index('--model_dir') + 1]
        if '--seed' in sys.argv:
            kwargs['seed'] = int(sys.argv[sys.argv.index('--seed') + 1])
        main_predict(**kwargs)

    else:
        print(f": {command}")
        print(": train, predict")
        sys.exit(1)
