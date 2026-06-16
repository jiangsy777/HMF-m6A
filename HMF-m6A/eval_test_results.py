"""
对 test_motif_results 验证集用训练好的融合模型+基序分类器预测,
绘制混淆矩阵和 ROC-AUC 曲线, 保存作图数据和图片.
不修改任何现有文件.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer
from typing import List
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')

from fusion_models import MultimodalFusionModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "test_motif_results")
TRUNK_PATH = os.path.join(BASE_DIR, "checkpoints", "multimodal_fusion_trunk.pt")
DNABERT3_PATH = os.path.join(BASE_DIR, "DNABERT3")
ROUTING_MAP_PATH = os.path.join(BASE_DIR, "classifiers", "routing_map.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "eval_test_results")

FILE_TO_MOTIF = {
    'AAACA': 'AAACA', 'AAACC': 'AAACC', 'AAACT': 'AAACT',
    'AGACA': 'AGACA', 'AGACC': 'AGACC', 'AGACT': 'AGACT',
    'GAACA': 'GAACA', 'GAACC': 'GAACC', 'GAACT': 'GAACT',
    'GGACA': 'GGACA', 'GGACC': 'GGACC', 'GGACT': 'GGACT',
    'TAACA': 'TAACA', 'TAACT': 'TAACT',
    'TGACA': 'TGACA', 'TGACC': 'TGACC', 'TGACT': 'TGACT',
    'other': 'others'
}

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
    unique_seqs = list(set(seqs))
    seq_to_struct = {}
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
        for _, row in df.iterrows():
            seq = normalize_sequence(str(row['text']).strip(), seq_len)
            sequences.append(seq)

        structures = batch_predict_secondary_structure(sequences, num_workers=4)

        self.processed_data = []
        for i in range(len(sequences)):
            seq = sequences[i]
            structure = structures[i]
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
                'str_features': struc
            })

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        d = self.processed_data[idx]
        return {
            'input_ids': d['input_ids'],
            'attention_mask': d['attention_mask'],
            'phy_features': torch.tensor(d['phy_features'], dtype=torch.float32),
            'str_features': torch.tensor(d['str_features'], dtype=torch.float32)
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'phy_features': torch.stack([item['phy_features'] for item in batch]),
        'str_features': torch.stack([item['str_features'] for item in batch])
    }


@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
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

    return np.concatenate(all_features, axis=0)


def predict_with_classifier(features, classifier, device, batch_size=256):
    classifier.eval()
    X = torch.tensor(features, dtype=torch.float32).to(device)
    all_probs = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        with torch.no_grad():
            probs = classifier(batch).cpu().numpy()
        all_probs.extend(probs)
    return np.array(all_probs)


def setup_matplotlib():
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'font.weight': 'normal',
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'mathtext.default': 'regular',
    })


def plot_confusion_matrix(cm, labels, save_prefix):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=11, width=0.5)
    cbar.outline.set_linewidth(0.5)

    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted label', fontsize=14)
    ax.set_ylabel('True label', fontsize=14)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14)

    ax.tick_params(axis='both', which='both', length=0)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    for fmt in ['svg', 'png']:
        fig.savefig(f'{save_prefix}.{fmt}', format=fmt)
    plt.close(fig)


def plot_roc_curve(fpr, tpr, auc_val, save_prefix, label_prefix=''):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.plot(fpr, tpr, color='#E64B35', lw=0.8,
            label=f'{label_prefix}AUC = {auc_val:.4f}')
    ax.plot([0, 1], [0, 1], color='grey', lw=0.5, linestyle='--')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.legend(loc='lower right', fontsize=12, frameon=True,
              edgecolor='black', fancybox=False)
    ax.set_aspect('equal')

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    for fmt in ['svg', 'png']:
        fig.savefig(f'{save_prefix}.{fmt}', format=fmt)
    plt.close(fig)


def plot_roc_curve_multi_group(group_results, save_prefix):
    fig, ax = plt.subplots(figsize=(6, 5.5))

    colors = plt.cm.tab20(np.linspace(0, 1, len(group_results)))

    for idx, r in enumerate(group_results):
        ax.plot(r['fpr'], r['tpr'], color=colors[idx], lw=0.8,
                label=f"{r['motif']} (AUC={r['auc']:.3f})")

    ax.plot([0, 1], [0, 1], color='grey', lw=0.5, linestyle='--')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.legend(loc='lower right', fontsize=8, frameon=True,
              edgecolor='black', fancybox=False, ncol=2)
    ax.set_aspect('equal')

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    for fmt in ['svg', 'png']:
        fig.savefig(f'{save_prefix}.{fmt}', format=fmt)
    plt.close(fig)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    setup_matplotlib()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  Evaluate on test_motif_results (Confusion Matrix + ROC-AUC)")
    print("=" * 70)
    print(f"  Device: {device}")

    print("\n[1] Loading fusion model...")
    checkpoint = torch.load(TRUNK_PATH, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = MultimodalFusionModel(
        model_name=DNABERT3_PATH,
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

    tokenizer = AutoTokenizer.from_pretrained(DNABERT3_PATH)

    print("\n[2] Loading motif classifiers...")
    with open(ROUTING_MAP_PATH, 'r') as f:
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

    print("\n[3] Predicting on test set...")
    all_labels = []
    all_probs = []
    all_preds = []
    all_motifs = []

    per_group_results = []

    for file_key, motif_name in sorted(FILE_TO_MOTIF.items()):
        test_path = os.path.join(TEST_DIR, f"test_{file_key}.csv")
        if not os.path.exists(test_path):
            print(f"  SKIP {file_key}: file not found")
            continue

        df = pd.read_csv(test_path)
        labels = df['label'].values.astype(int)
        pos = int(labels.sum())
        neg = int(len(labels) - pos)
        print(f"\n  {file_key}: {len(df)} samples (pos:{pos} neg:{neg})")

        print(f"    Extracting features...")
        dataset = FeatureExtractionDataset(df, tokenizer, seq_len=201, max_len=256)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False,
                                collate_fn=collate_fn, num_workers=0,
                                pin_memory=device.type == 'cuda')
        features = extract_features(model, dataloader, device)
        print(f"    Features: {features.shape}")

        if motif_name not in classifiers:
            print(f"    WARNING: No classifier for {motif_name}, skipping")
            continue

        clf = classifiers[motif_name]
        probs = predict_with_classifier(features, clf, device)
        preds = (probs > 0.5).astype(int)

        all_labels.extend(labels)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_motifs.extend([file_key] * len(labels))

        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.5

        cm = confusion_matrix(labels, preds)
        print(f"    AUC={auc:.4f}")
        print(f"    CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

        fpr, tpr, _ = roc_curve(labels, probs)

        per_group_results.append({
            'motif': file_key,
            'auc': auc,
            'cm': cm,
            'fpr': fpr,
            'tpr': tpr,
            'labels': labels,
            'probs': probs,
            'preds': preds
        })

        cm_path_prefix = os.path.join(OUTPUT_DIR, f"cm_{file_key}")
        plot_confusion_matrix(cm, ['Negative', 'Positive'], cm_path_prefix)

        roc_path_prefix = os.path.join(OUTPUT_DIR, f"roc_{file_key}")
        plot_roc_curve(fpr, tpr, auc, roc_path_prefix, label_prefix=f'{file_key} ')

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    print(f"\n{'='*70}")
    print(f"  Overall: {len(all_labels)} samples")
    print(f"{'='*70}")

    overall_auc = roc_auc_score(all_labels, all_probs)
    overall_cm = confusion_matrix(all_labels, all_preds)
    print(f"  AUC: {overall_auc:.4f}")
    print(f"  CM: TN={overall_cm[0,0]} FP={overall_cm[0,1]} FN={overall_cm[1,0]} TP={overall_cm[1,1]}")

    overall_fpr, overall_tpr, overall_thresholds = roc_curve(all_labels, all_probs)

    cm_overall_prefix = os.path.join(OUTPUT_DIR, "cm_overall")
    plot_confusion_matrix(overall_cm, ['Negative', 'Positive'], cm_overall_prefix)

    roc_overall_prefix = os.path.join(OUTPUT_DIR, "roc_overall")
    plot_roc_curve(overall_fpr, overall_tpr, overall_auc, roc_overall_prefix)

    roc_multi_prefix = os.path.join(OUTPUT_DIR, "roc_all_groups")
    plot_roc_curve_multi_group(per_group_results, roc_multi_prefix)

    print("\n[4] Saving plot data...")

    overall_roc_data = pd.DataFrame({
        'FPR': overall_fpr,
        'TPR': overall_tpr,
        'Threshold': overall_thresholds
    })
    overall_roc_data.to_csv(os.path.join(OUTPUT_DIR, "roc_overall_data.csv"), index=False)

    per_group_roc_data = []
    for r in per_group_results:
        for i in range(len(r['fpr'])):
            per_group_roc_data.append({
                'motif': r['motif'],
                'FPR': r['fpr'][i],
                'TPR': r['tpr'][i]
            })
    pd.DataFrame(per_group_roc_data).to_csv(
        os.path.join(OUTPUT_DIR, "roc_per_group_data.csv"), index=False)

    cm_data_rows = []
    cm_data_rows.append({
        'group': 'overall',
        'TN': int(overall_cm[0, 0]),
        'FP': int(overall_cm[0, 1]),
        'FN': int(overall_cm[1, 0]),
        'TP': int(overall_cm[1, 1]),
        'AUC': round(overall_auc, 4)
    })
    for r in per_group_results:
        cm_data_rows.append({
            'group': r['motif'],
            'TN': int(r['cm'][0, 0]),
            'FP': int(r['cm'][0, 1]),
            'FN': int(r['cm'][1, 0]),
            'TP': int(r['cm'][1, 1]),
            'AUC': round(r['auc'], 4)
        })
    pd.DataFrame(cm_data_rows).to_csv(
        os.path.join(OUTPUT_DIR, "confusion_matrix_data.csv"), index=False)

    pred_detail = pd.DataFrame({
        'motif': all_motifs,
        'true_label': all_labels,
        'pred_prob': all_probs,
        'pred_label': all_preds
    })
    pred_detail.to_csv(os.path.join(OUTPUT_DIR, "all_predictions.csv"), index=False)

    print(f"\n  Results saved to: {OUTPUT_DIR}/")
    print(f"    - cm_overall.svg/png        (Overall confusion matrix)")
    print(f"    - roc_overall.svg/png       (Overall ROC curve)")
    print(f"    - roc_all_groups.svg/png    (Per-group ROC curves)")
    print(f"    - cm_{{group}}.svg/png       (Per-group confusion matrix)")
    print(f"    - roc_{{group}}.svg/png      (Per-group ROC curve)")
    print(f"    - roc_overall_data.csv      (Overall ROC data)")
    print(f"    - roc_per_group_data.csv    (Per-group ROC data)")
    print(f"    - confusion_matrix_data.csv (All confusion matrices)")
    print(f"    - all_predictions.csv       (All predictions)")
    print("=" * 70)


if __name__ == "__main__":
    main()
