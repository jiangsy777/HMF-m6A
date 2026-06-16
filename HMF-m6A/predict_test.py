"""
对独立验证集 test_motif_results 进行预测
步骤1: 用融合模型提取1536维特征
步骤2: 用基序分类器做预测，计算混淆矩阵、Accuracy、F1等
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)
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


def batch_predict_secondary_structure(seqs: List[str], num_workers: int = 4) -> List[str]:
    import RNA
    total = len(seqs)
    unique_seqs = list(set(seqs))
    seq_to_struct = {}

    print(f"    Structure: {len(unique_seqs)} unique / {total} total")

    chunk_size = max(1, len(unique_seqs) // (num_workers * 4))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_fold_single, unique_seqs, chunksize=chunk_size))

    for seq, struct in zip(unique_seqs, results):
        seq_to_struct[seq] = struct

    return [seq_to_struct[seq] for seq in seqs]


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


class FeatureExtractionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, seq_len: int = 201, max_len: int = 256):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_len = max_len

        sequences = []
        labels = []
        for _, row in df.iterrows():
            seq = normalize_sequence(str(row['text']).strip(), seq_len)
            sequences.append(seq)
            labels.append(int(row['label']))

        structures = batch_predict_secondary_structure(sequences, num_workers=4)

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
def extract_features_for_group(model, df, tokenizer, device, batch_size=16):
    dataset = FeatureExtractionDataset(df, tokenizer, seq_len=201, max_len=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    model.eval()
    all_features = []
    all_labels = []

    use_amp = device.type == 'cuda'
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        if use_amp:
            with autocast(device_type='cuda'):
                outputs = model(batch)
        else:
            outputs = model(batch)

        all_features.append(outputs['F_combined'].cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)
    return features, labels


def predict_with_classifier(features, labels, classifier, device, batch_size=256):
    classifier.eval()
    X = torch.tensor(features, dtype=torch.float32).to(device)

    all_probs = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        with torch.no_grad():
            probs = classifier(batch).cpu().numpy()
        all_probs.extend(probs)

    probs = np.array(all_probs)
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    return {
        'accuracy': acc, 'auc': auc,
        'precision': precision, 'recall': recall, 'f1': f1,
        'cm': cm, 'probs': probs, 'preds': preds
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- 1. 加载融合模型 ----
    print("\n[1] Loading fusion model...")
    checkpoint = torch.load(
        "./checkpoints/multimodal_fusion_trunk.pt",
        map_location=device, weights_only=False
    )
    config = checkpoint['config']
    print(f"  Best training AUC: {checkpoint.get('best_auc', 'N/A'):.4f}")

    model = MultimodalFusionModel(
        model_name="./DNABERT3",
        embedding_dim=config['EMBEDDING_DIM'],
        phy_input_dim=config['PHY_FEATURE_DIM'],
        phy_channels=config['CNN_CHANNELS_PHY'],
        str_input_dim=config['STR_FEATURE_DIM'],
        str_channels=config['CNN_CHANNELS_STR'],
        kernel_size=config['KERNEL_SIZE'],
        recon_phy_channels=config['RECON_PHY_CHANNELS'],
        recon_str_channels=config['RECON_STR_CHANNELS'],
        recon_target_length=config['RECON_TARGET_LENGTH'],
        classifier_hidden=config['CLASSIFIER_HIDDEN_DIM'],
        dropout=config['DROPOUT']
    ).to(device)

    model.load_fusion_trunk_state(checkpoint['fusion_trunk'])
    print("  Fusion trunk loaded")

    tokenizer = AutoTokenizer.from_pretrained("./DNABERT3")

    # ---- 2. 加载路由映射和基序分类器 ----
    print("\n[2] Loading motif classifiers...")
    with open("./classifiers/routing_map.json", 'r') as f:
        routing_info = json.load(f)

    routing = routing_info['routing']
    hidden_dims = routing_info['hidden_dims']
    dropout = routing_info['dropout']

    classifiers = {}
    for motif_name, weight_path in routing.items():
        ckpt = torch.load(weight_path, map_location=device, weights_only=False)
        clf = MotifClassifier(
            input_dim=1536, hidden_dims=hidden_dims, dropout=dropout
        ).to(device)
        clf.load_state_dict(ckpt['model_state_dict'])
        clf.eval()
        classifiers[motif_name] = clf
    print(f"  Loaded {len(classifiers)} classifiers")

    # ---- 3. 基序名映射 ----
    file_to_motif = {
        'AAACA': 'AAACA', 'AAACC': 'AAACC', 'AAACT': 'AAACT',
        'AGACA': 'AGACA', 'AGACC': 'AGACC', 'AGACT': 'AGACT',
        'GAACA': 'GAACA', 'GAACC': 'GAACC', 'GAACT': 'GAACT',
        'GGACA': 'GGACA', 'GGACC': 'GGACC', 'GGACT': 'GGACT',
        'TAACA': 'TAACA', 'TAACT': 'TAACT',
        'TGACA': 'TGACA', 'TGACC': 'TGACC', 'TGACT': 'TGACT',
        'other': 'others'
    }

    # ---- 4. 逐组预测 ----
    print("\n[3] Predicting on independent test set...")
    test_dir = "./test_motif_results"
    all_results = []

    overall_labels = []
    overall_preds = []
    overall_probs = []

    for file_key, motif_name in file_to_motif.items():
        test_path = os.path.join(test_dir, f"test_{file_key}.csv")
        if not os.path.exists(test_path):
            print(f"\n  SKIP {file_key}: file not found")
            continue

        df = pd.read_csv(test_path)
        pos = int((df['label'] == 1).sum())
        neg = int((df['label'] == 0).sum())
        print(f"\n{'='*60}")
        print(f"  Motif: {file_key} -> classifier: {motif_name}")
        print(f"  Samples: {len(df)} (pos:{pos} neg:{neg})")
        print(f"{'='*60}")

        print(f"  Extracting 1536-dim features...")
        features, labels = extract_features_for_group(
            model, df, tokenizer, device, batch_size=16
        )
        print(f"  Features: {features.shape}")

        if motif_name not in classifiers:
            print(f"  WARNING: No classifier for {motif_name}, skipping")
            continue

        clf = classifiers[motif_name]
        results = predict_with_classifier(features, labels, clf, device)

        cm = results['cm']
        print(f"\n  === Results for {file_key} ===")
        print(f"  AUC:       {results['auc']:.4f}")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1:        {results['f1']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"           Pred_Neg  Pred_Pos")
        print(f"  Actual_Neg  {cm[0,0]:>6}   {cm[0,1]:>6}")
        print(f"  Actual_Pos  {cm[1,0]:>6}   {cm[1,1]:>6}")

        all_results.append({
            'motif': file_key,
            'classifier': motif_name,
            'samples': len(df),
            'pos': pos,
            'neg': neg,
            'auc': results['auc'],
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        })

        overall_labels.extend(labels)
        overall_preds.extend(results['preds'])
        overall_probs.extend(results['probs'])

    # ---- 5. 汇总 ----
    print(f"\n{'='*70}")
    print("  OVERALL RESULTS ON INDEPENDENT TEST SET")
    print(f"{'='*70}")

    overall_labels = np.array(overall_labels)
    overall_preds = np.array(overall_preds)
    overall_probs = np.array(overall_probs)

    overall_acc = accuracy_score(overall_labels, overall_preds)
    try:
        overall_auc = roc_auc_score(overall_labels, overall_probs)
    except ValueError:
        overall_auc = 0.5
    overall_p, overall_r, overall_f1, _ = precision_recall_fscore_support(
        overall_labels, overall_preds, average='binary', zero_division=0
    )
    overall_cm = confusion_matrix(overall_labels, overall_preds)

    print(f"  Total samples: {len(overall_labels)}")
    print(f"  AUC:       {overall_auc:.4f}")
    print(f"  Accuracy:  {overall_acc:.4f}")
    print(f"  Precision: {overall_p:.4f}")
    print(f"  Recall:    {overall_r:.4f}")
    print(f"  F1:        {overall_f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"           Pred_Neg  Pred_Pos")
    print(f"  Actual_Neg  {overall_cm[0,0]:>6}   {overall_cm[0,1]:>6}")
    print(f"  Actual_Pos  {overall_cm[1,0]:>6}   {overall_cm[1,1]:>6}")

    print(f"\n{'='*70}")
    print("  PER-MOTIF SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Motif':<10} {'N':>6} {'Pos':>5} {'Neg':>5} {'AUC':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'TN':>5} {'FP':>5} {'FN':>5} {'TP':>5}")
    print(f"  {'-'*96}")
    for r in all_results:
        print(f"  {r['motif']:<10} {r['samples']:>6} {r['pos']:>5} {r['neg']:>5} "
              f"{r['auc']:>7.4f} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['f1']:>7.4f} "
              f"{r['tn']:>5} {r['fp']:>5} {r['fn']:>5} {r['tp']:>5}")

    if all_results:
        aucs = [r['auc'] for r in all_results]
        accs = [r['accuracy'] for r in all_results]
        f1s = [r['f1'] for r in all_results]
        print(f"\n  Mean AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
        print(f"  Mean Acc: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
        print(f"  Mean F1:  {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")

    df_results = pd.DataFrame(all_results)
    df_results.to_csv("./test_motif_results_summary.csv", index=False)
    print(f"\n  Detailed results saved to: ./test_motif_results_summary.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
