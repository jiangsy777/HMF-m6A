"""
HMF 多模态融合模块训练脚本
训练 CNN 提取器、稀疏门控机制、重构解码器
损失 = BCE分类损失 + alpha * MSE重构损失
"""

import os
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from typing import List
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from fusion_models import MultimodalFusionModel


NUCLEOTIDE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4}
EIIP_DICT = {'A': 0.1260, 'C': 0.1340, 'G': 0.0806, 'T': 0.1335, 'U': 0.1335, 'N': 0.0}
DPP_DIM = 16
STRUCTURE_TO_IDX = {'.': 0, '(': 1, ')': 2, 'N': 3}


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
    import RNA
    total = len(seqs)
    unique_seqs = list(set(seqs))
    seq_to_struct = {}

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


def extract_5mer(seq: str, center_pos: int = 100) -> str:
    seq = seq.upper().replace('T', 'U')
    half = 2
    return seq[center_pos - half: center_pos + half + 1]


def normalize_sequence(seq: str, target_len: int = 201) -> str:
    seq = seq.upper().replace('T', 'U')
    if len(seq) < target_len:
        seq = seq + 'N' * (target_len - len(seq))
    elif len(seq) > target_len:
        mid = len(seq) // 2
        half = target_len // 2
        seq = seq[mid - half: mid + half + 1]
    return seq


class Config:
    DATA_PATH = "all_train_samples.tsv"
    MODEL_NAME = "./DNABERT3"
    SEQ_LEN = 201
    MAX_LEN = 256

    PHY_FEATURE_DIM = 29
    STR_FEATURE_DIM = 8
    EMBEDDING_DIM = 768

    CNN_CHANNELS_PHY = [128, 256, 512]
    CNN_CHANNELS_STR = [64, 128, 256]
    KERNEL_SIZE = 3

    RECON_PHY_CHANNELS = [256, 512]
    RECON_STR_CHANNELS = [128, 256]
    RECON_TARGET_LENGTH = 25

    CLASSIFIER_INPUT_DIM = 1536
    CLASSIFIER_HIDDEN_DIM = 256
    DROPOUT = 0.3

    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.01
    ALPHA_RECON = 0.1
    PATIENCE = 5

    NUM_WORKERS_RNA = 4

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True

    OUTPUT_DIR = "./checkpoints"
    TRUNK_SAVE_PATH = "./checkpoints/multimodal_fusion_trunk.pt"

    CACHE_DIR = "./cache"


def get_cache_path(data_hash: str, split: str) -> str:
    return os.path.join(Config.CACHE_DIR, f"{split}_{data_hash}.pt")


def compute_data_hash(df: pd.DataFrame) -> str:
    content = f"{len(df)}_{df['label'].sum()}_{hashlib.md5(str(df['text'].values[:10]).encode()).hexdigest()[:8]}"
    return content


class M6AFusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, seq_len: int = 201,
                 max_len: int = 256, split: str = "train", cache_dir: str = None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_len = max_len

        print("Preprocessing data...")

        print("  [1/3] Extracting sequences and labels...")
        sequences = []
        labels = []
        for _, row in self.df.iterrows():
            seq = normalize_sequence(str(row['text']).strip(), seq_len)
            sequences.append(seq)
            labels.append(int(row['label']))

        print(f"  [2/3] Batch predicting secondary structure ({len(sequences)} seqs)...")
        structures = batch_predict_secondary_structure(
            sequences, verbose=True, num_workers=Config.NUM_WORKERS_RNA
        )

        print("  [3/3] Computing features + DNABERT tokenization...")
        self.processed_data = []
        for i in range(len(sequences)):
            seq = sequences[i]
            structure = structures[i]
            label = labels[i]

            phy_features = calculate_physicochemical_features(seq).astype(np.float32)
            str_features = encode_secondary_structure(structure, seq_len).astype(np.float32)

            kmer_text = " ".join([seq[j:j + 3] for j in range(len(seq) - 2)])
            encoded = self.tokenizer(
                kmer_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )

            self.processed_data.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'phy_features': phy_features,
                'str_features': str_features,
                'label': label
            })

            if (i + 1) % 10000 == 0:
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


def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'phy_features': torch.stack([item['phy_features'] for item in batch]),
        'str_features': torch.stack([item['str_features'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch])
    }


def train_epoch(model, dataloader, optimizer, scheduler, scaler, config, epoch):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_recon_loss = 0.0
    all_preds = []
    all_labels = []

    bce_fn = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        optimizer.zero_grad()

        use_amp = config.USE_AMP and config.DEVICE.type == 'cuda'

        if use_amp:
            with autocast(device_type='cuda'):
                outputs = model(batch)
                cls_loss = bce_fn(outputs['pred_logits'], batch['labels'])
                recon_loss = (mse_fn(outputs['recon_phy'], outputs['phy_feature_map']) +
                              mse_fn(outputs['recon_str'], outputs['str_feature_map']))
                loss = cls_loss + config.ALPHA_RECON * recon_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch)
            cls_loss = bce_fn(outputs['pred_logits'], batch['labels'])
            recon_loss = (mse_fn(outputs['recon_phy'], outputs['phy_feature_map']) +
                          mse_fn(outputs['recon_str'], outputs['str_feature_map']))
            loss = cls_loss + config.ALPHA_RECON * recon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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

        if (step + 1) % 100 == 0:
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
def evaluate(model, dataloader, config):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    bce_fn = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()
    use_amp = config.USE_AMP and config.DEVICE.type == 'cuda'

    for batch in dataloader:
        batch = {k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        if use_amp:
            with autocast(device_type='cuda'):
                outputs = model(batch)
                cls_loss = bce_fn(outputs['pred_logits'], batch['labels'])
                recon_loss = (mse_fn(outputs['recon_phy'], outputs['phy_feature_map']) +
                              mse_fn(outputs['recon_str'], outputs['str_feature_map']))
                loss = cls_loss + config.ALPHA_RECON * recon_loss
        else:
            outputs = model(batch)
            cls_loss = bce_fn(outputs['pred_logits'], batch['labels'])
            recon_loss = (mse_fn(outputs['recon_phy'], outputs['phy_feature_map']) +
                          mse_fn(outputs['recon_str'], outputs['str_feature_map']))
            loss = cls_loss + config.ALPHA_RECON * recon_loss

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
        all_labels, preds, average='binary'
    )

    return {
        'loss': avg_loss, 'accuracy': acc, 'auc': auc,
        'precision': precision, 'recall': recall, 'f1': f1
    }


def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  HMF Multimodal Fusion Training")
    print("  SeqLen: 201nt | DNABERT: Frozen | Loss: BCE + alpha*MSE_recon")
    print("=" * 70)
    print(f"Device: {config.DEVICE}, AMP: {config.USE_AMP}")

    print("\n[1] Loading data...")
    if config.DATA_PATH.endswith('.tsv'):
        df = pd.read_csv(config.DATA_PATH, sep='\t')
    else:
        df = pd.read_csv(config.DATA_PATH)
    print(f"Dataset: {len(df)} samples (pos:{(df['label'] == 1).sum()} / neg:{(df['label'] == 0).sum()})")

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    print("\n[2] Loading DNABERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    print("\n[3] Creating datasets...")
    train_dataset = M6AFusionDataset(
        train_df, tokenizer, config.SEQ_LEN, config.MAX_LEN,
        split="train", cache_dir=config.CACHE_DIR
    )
    val_dataset = M6AFusionDataset(
        val_df, tokenizer, config.SEQ_LEN, config.MAX_LEN,
        split="val", cache_dir=config.CACHE_DIR
    )

    pin_mem = config.DEVICE.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=pin_mem)

    print("\n[4] Initializing model...")
    model = MultimodalFusionModel(
        model_name=config.MODEL_NAME,
        embedding_dim=config.EMBEDDING_DIM,
        phy_input_dim=config.PHY_FEATURE_DIM,
        phy_channels=config.CNN_CHANNELS_PHY,
        str_input_dim=config.STR_FEATURE_DIM,
        str_channels=config.CNN_CHANNELS_STR,
        kernel_size=config.KERNEL_SIZE,
        recon_phy_channels=config.RECON_PHY_CHANNELS,
        recon_str_channels=config.RECON_STR_CHANNELS,
        recon_target_length=config.RECON_TARGET_LENGTH,
        classifier_hidden=config.CLASSIFIER_HIDDEN_DIM,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    for param in model.bert_extractor.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,} "
          f"({trainable_params / total_params * 100:.2f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader) * 5, T_mult=2
    )
    scaler = GradScaler('cuda') if (config.USE_AMP and config.DEVICE.type == 'cuda') else None

    print("\n[5] Training...")
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch}/{config.EPOCHS}")
        print(f"{'=' * 40}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, config, epoch
        )
        val_metrics = evaluate(model, val_loader, config)

        sparsity_phy = model.gate_phy.element_linear.get_sparsity()
        sparsity_str = model.gate_str.element_linear.get_sparsity()

        print(f"\n[Epoch {epoch}] Train: Loss={train_metrics['loss']:.4f} "
              f"Cls={train_metrics['cls_loss']:.4f} "
              f"Recon={train_metrics['recon_loss']:.4f} "
              f"AUC={train_metrics['auc']:.4f} Acc={train_metrics['accuracy']:.4f}")
        print(f"[Epoch {epoch}] Val:   Loss={val_metrics['loss']:.4f} "
              f"AUC={val_metrics['auc']:.4f} Acc={val_metrics['accuracy']:.4f} "
              f"Precision={val_metrics['precision']:.4f} Recall={val_metrics['recall']:.4f} "
              f"F1={val_metrics['f1']:.4f}")
        print(f"[Epoch {epoch}] Gate sparsity: Phy={sparsity_phy:.4f} Str={sparsity_str:.4f}")

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0

            trunk_state = model.get_fusion_trunk_state()
            torch.save({
                'fusion_trunk': trunk_state,
                'epoch': epoch,
                'best_auc': best_auc,
                'config': {
                    'EMBEDDING_DIM': config.EMBEDDING_DIM,
                    'PHY_FEATURE_DIM': config.PHY_FEATURE_DIM,
                    'STR_FEATURE_DIM': config.STR_FEATURE_DIM,
                    'CNN_CHANNELS_PHY': config.CNN_CHANNELS_PHY,
                    'CNN_CHANNELS_STR': config.CNN_CHANNELS_STR,
                    'KERNEL_SIZE': config.KERNEL_SIZE,
                    'RECON_PHY_CHANNELS': config.RECON_PHY_CHANNELS,
                    'RECON_STR_CHANNELS': config.RECON_STR_CHANNELS,
                    'RECON_TARGET_LENGTH': config.RECON_TARGET_LENGTH,
                    'CLASSIFIER_HIDDEN_DIM': config.CLASSIFIER_HIDDEN_DIM,
                    'DROPOUT': config.DROPOUT,
                }
            }, config.TRUNK_SAVE_PATH)
            print(f"  * New best model! AUC: {best_auc:.4f}, trunk saved")
        else:
            patience_counter += 1
            print(f"  Val AUC not improved ({patience_counter}/{config.PATIENCE})")

        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping! Best AUC: {best_auc:.4f}")
            break

    print("\n" + "=" * 70)
    print(f"Training complete! Best val AUC: {best_auc:.4f}")
    print(f"Fusion trunk saved: {config.TRUNK_SAVE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
