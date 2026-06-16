"""
HMF 特征提取与基序分组脚本
1. 加载 DNABERT (冻结) + 训练好的融合主干权重
2. 推理得到 1536 维 F_combined，保存为 .npz
3. 按正样本基序频次分组，保存各组 train.tsv / val.tsv
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer
from collections import Counter, defaultdict
from typing import List
import warnings
warnings.filterwarnings('ignore')

from fusion_models import MultimodalFusionModel
from motif_grouper import extract_5mer, analyze_motifs_by_positive, group_data_by_motif


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

    from concurrent.futures import ProcessPoolExecutor
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


class Config:
    DATA_PATH = "all_train_samples.tsv"
    MODEL_NAME = "./DNABERT3"
    TRUNK_PATH = "./checkpoints/multimodal_fusion_trunk.pt"
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
    BATCH_SIZE = 32
    MOTIF_THRESHOLD_RATIO = 0.015

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True

    FEATURES_SAVE_PATH = "./features/m6a_features.npz"
    GROUPED_DATA_DIR = "./grouped_data"


class FeatureExtractionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, seq_len: int = 201, max_len: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_len = max_len

        print("Preprocessing for feature extraction...")
        sequences = []
        labels = []
        for _, row in self.df.iterrows():
            seq = normalize_sequence(str(row['text']).strip(), seq_len)
            sequences.append(seq)
            labels.append(int(row['label']))

        print(f"  Batch predicting secondary structure ({len(sequences)} seqs)...")
        structures = batch_predict_secondary_structure(sequences, verbose=True)

        print("  Computing features + tokenization...")
        self.processed_data = []
        for i in range(len(sequences)):
            seq = sequences[i]
            structure = structures[i]
            label = labels[i]

            phy = calculate_physicochemical_features(seq).astype(np.float32)
            struc = encode_secondary_structure(structure, seq_len).astype(np.float32)

            kmer_text = " ".join([seq[j:j + 3] for j in range(len(seq) - 2)])
            encoded = self.tokenizer(
                kmer_text, padding='max_length', truncation=True,
                max_length=self.max_len, return_tensors='pt'
            )

            self.processed_data.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'phy_features': phy,
                'str_features': struc,
                'label': label
            })

        print(f"  Preprocessing done: {len(self.processed_data)} samples")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        d = self.processed_data[idx]
        return {
            'input_ids': d['input_ids'],
            'attention_mask': d['attention_mask'],
            'phy_features': torch.tensor(d['phy_features'], dtype=torch.float32),
            'str_features': torch.tensor(d['str_features'], dtype=torch.float32),
            'label': torch.tensor(d['label'], dtype=torch.float32)
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'phy_features': torch.stack([item['phy_features'] for item in batch]),
        'str_features': torch.stack([item['str_features'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch])
    }


@torch.no_grad()
def extract_features(model, dataloader, config):
    model.eval()
    all_features = []
    all_labels = []

    use_amp = config.USE_AMP and config.DEVICE.type == 'cuda'

    for batch in dataloader:
        batch = {k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        if use_amp:
            with autocast(device_type='cuda'):
                outputs = model(batch)
        else:
            outputs = model(batch)

        f_combined = outputs['F_combined'].cpu().numpy()
        labels = batch['labels'].cpu().numpy()

        all_features.append(f_combined)
        all_labels.extend(labels)

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)
    return features, labels


def main():
    config = Config()
    os.makedirs(os.path.dirname(config.FEATURES_SAVE_PATH), exist_ok=True)
    os.makedirs(config.GROUPED_DATA_DIR, exist_ok=True)

    print("=" * 70)
    print("  HMF Feature Extraction & Motif Grouping")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")

    print("\n[1] Loading data...")
    if config.DATA_PATH.endswith('.tsv'):
        df = pd.read_csv(config.DATA_PATH, sep='\t')
    else:
        df = pd.read_csv(config.DATA_PATH)
    print(f"Dataset: {len(df)} samples (pos:{(df['label'] == 1).sum()} / neg:{(df['label'] == 0).sum()})")

    print("\n[2] Loading model with fusion trunk...")
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

    if os.path.exists(config.TRUNK_PATH):
        checkpoint = torch.load(config.TRUNK_PATH, map_location=config.DEVICE, weights_only=False)
        model.load_fusion_trunk_state(checkpoint['fusion_trunk'])
        print(f"  Loaded trunk: {config.TRUNK_PATH} (epoch={checkpoint.get('epoch', '?')}, "
              f"AUC={checkpoint.get('best_auc', '?')})")
    else:
        print(f"  WARNING: {config.TRUNK_PATH} not found, using random trunk!")

    print("\n[3] Extracting 1536-dim features...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    dataset = FeatureExtractionDataset(df, tokenizer, config.SEQ_LEN, config.MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0,
                            pin_memory=config.DEVICE.type == 'cuda')

    features, labels = extract_features(model, dataloader, config)
    print(f"  Feature matrix: {features.shape} (expected: ({len(df)}, 1536))")

    motifs = []
    sequences = []
    for _, row in df.iterrows():
        seq = str(row['text']).strip()
        motifs.append(extract_5mer(seq))
        sequences.append(seq)

    print("\n[4] Saving features...")
    np.savez(config.FEATURES_SAVE_PATH,
             features=features,
             labels=labels,
             motifs=np.array(motifs),
             sequences=np.array(sequences))
    print(f"  Saved: {config.FEATURES_SAVE_PATH}")

    print("\n[5] Motif grouping (threshold = num_positive * 0.015)...")
    df_with_motif, major_motifs, has_others = analyze_motifs_by_positive(
        df, threshold_ratio=config.MOTIF_THRESHOLD_RATIO
    )

    group_names = group_data_by_motif(
        df_with_motif, major_motifs, has_others, config.GROUPED_DATA_DIR
    )

    motif_to_indices = defaultdict(list)
    for idx, row in df_with_motif.iterrows():
        motif = row['motif']
        if motif in major_motifs:
            group_key = motif
        else:
            group_key = 'others'
        motif_to_indices[group_key].append(idx)

    index_map_path = os.path.join(config.GROUPED_DATA_DIR, 'feature_index_map.json')
    with open(index_map_path, 'w') as f:
        json.dump({k: v for k, v in motif_to_indices.items()}, f, indent=2)
    print(f"  Feature index map saved: {index_map_path}")

    print("\n" + "=" * 70)
    print("  Feature extraction & motif grouping complete!")
    print(f"  Features: {config.FEATURES_SAVE_PATH}")
    print(f"  Groups: {config.GROUPED_DATA_DIR}")
    print(f"  Num groups: {len(group_names)} (with {'others' if has_others else 'no'} others)")
    print("=" * 70)


if __name__ == "__main__":
    main()
