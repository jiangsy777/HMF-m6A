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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             v_measure_score, silhouette_score,
                             calinski_harabasz_score, davies_bouldin_score)
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
OUTPUT_DIR = os.path.join(BASE_DIR, "dim_reduction_results")

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
def extract_features_with_bert(model, dataloader, device):
    model.eval()
    all_f_bert = []
    all_f_combined = []
    use_amp = device.type == 'cuda'

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        if use_amp:
            with autocast(device_type='cuda'):
                outputs = model(batch)
        else:
            outputs = model(batch)
        all_f_bert.append(outputs['F_bert'].cpu().numpy())
        all_f_combined.append(outputs['F_combined'].cpu().numpy())

    f_bert = np.concatenate(all_f_bert, axis=0)
    f_combined = np.concatenate(all_f_combined, axis=0)
    return f_bert, f_combined


def extract_fc_vector(classifier, features, device, batch_size=256):
    classifier.eval()
    X = torch.tensor(features, dtype=torch.float32).to(device)
    all_vectors = []

    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        with torch.no_grad():
            x = batch
            for layer_idx in range(8):
                x = classifier.classifier[layer_idx](x)
            all_vectors.append(x.cpu().numpy())

    return np.concatenate(all_vectors, axis=0)


def compute_metrics(X_2d, labels, seed=42):
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=seed, batch_size=256, n_init=10)
    pred_clusters = kmeans.fit_predict(X_2d)

    ari = adjusted_rand_score(labels, pred_clusters)
    nmi = normalized_mutual_info_score(labels, pred_clusters)
    v_meas = v_measure_score(labels, pred_clusters)

    sil = silhouette_score(X_2d, labels, sample_size=min(10000, len(labels)),
                           random_state=seed)
    ch = calinski_harabasz_score(X_2d, labels)
    db = davies_bouldin_score(X_2d, labels)

    return {
        'ARI': ari,
        'NMI': nmi,
        'V-measure': v_meas,
        'Silhouette': sil,
        'Calinski-Harabasz': ch,
        'Davies-Bouldin': db
    }


def select_best_metric(metrics_bert, metrics_fc):
    higher_better = ['ARI', 'NMI', 'V-measure', 'Silhouette', 'Calinski-Harabasz']
    lower_better = ['Davies-Bouldin']

    best_name = None
    best_improvement = -np.inf

    for name in higher_better:
        val_bert = metrics_bert[name]
        val_fc = metrics_fc[name]
        if val_bert > 1e-10:
            improvement = (val_fc - val_bert) / abs(val_bert)
        else:
            improvement = val_fc - val_bert
        if improvement > best_improvement:
            best_improvement = improvement
            best_name = name

    for name in lower_better:
        val_bert = metrics_bert[name]
        val_fc = metrics_fc[name]
        if val_bert > 1e-10:
            improvement = (val_bert - val_fc) / abs(val_bert)
        else:
            improvement = val_bert - val_fc
        if improvement > best_improvement:
            best_improvement = improvement
            best_name = name

    return best_name


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


COLOR_POS = '#E64B35'
COLOR_NEG = '#4DBBD5'
MARKER_SIZE = 4
ALPHA = 0.45


def plot_single_scatter(coords_2d, labels, metrics_dict, highlight_metric,
                        xlabel, ylabel, save_prefix):
    fig, ax = plt.subplots(figsize=(5, 4.8))

    neg_mask = labels == 0
    pos_mask = labels == 1

    ax.scatter(coords_2d[pos_mask, 0], coords_2d[pos_mask, 1],
               c=COLOR_POS, s=MARKER_SIZE, alpha=ALPHA, label='Positive',
               edgecolors='none', rasterized=True)
    ax.scatter(coords_2d[neg_mask, 0], coords_2d[neg_mask, 1],
               c=COLOR_NEG, s=MARKER_SIZE, alpha=ALPHA, label='Negative',
               edgecolors='none', rasterized=True)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')

    metric_val = metrics_dict[highlight_metric]
    ax.text(0.97, 0.03, f'{highlight_metric} = {metric_val:.4f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='black', linewidth=0.5, alpha=0.9))

    ax.legend(loc='upper left', fontsize=11, frameon=True,
              edgecolor='black', fancybox=False, markerscale=3)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    for fmt in ['svg', 'png']:
        fig.savefig(f'{save_prefix}.{fmt}', format=fmt)
    plt.close(fig)


def plot_combined_2x2(coords_dict, labels, metrics_dict, highlight_metric, save_prefix):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10))

    panel_info = [
        ('bert_tsne', 'DNABERT3 + t-SNE', 't-SNE 1', 't-SNE 2'),
        ('bert_pca', 'DNABERT3 + PCA', 'PC 1', 'PC 2'),
        ('fc_tsne', 'Motif Classifier + t-SNE', 't-SNE 1', 't-SNE 2'),
        ('fc_pca', 'Motif Classifier + PCA', 'PC 1', 'PC 2'),
    ]

    neg_mask = labels == 0
    pos_mask = labels == 1

    for idx, (key, panel_label, xlabel, ylabel) in enumerate(panel_info):
        ax = axes[idx // 2, idx % 2]
        coords = coords_dict[key]
        m = metrics_dict[key]

        ax.scatter(coords[pos_mask, 0], coords[pos_mask, 1],
                   c=COLOR_POS, s=MARKER_SIZE, alpha=ALPHA, label='Positive',
                   edgecolors='none', rasterized=True)
        ax.scatter(coords[neg_mask, 0], coords[neg_mask, 1],
                   c=COLOR_NEG, s=MARKER_SIZE, alpha=ALPHA, label='Negative',
                   edgecolors='none', rasterized=True)

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')

        metric_val = m[highlight_metric]
        ax.text(0.97, 0.03, f'{highlight_metric} = {metric_val:.4f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='black', linewidth=0.5, alpha=0.9))

        ax.text(0.03, 0.97, panel_label,
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='black', linewidth=0.5, alpha=0.9))

        if idx == 0:
            ax.legend(loc='lower left', fontsize=10, frameon=True,
                      edgecolor='black', fancybox=False, markerscale=3)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    plt.tight_layout(pad=1.5)

    for fmt in ['svg', 'png']:
        fig.savefig(f'{save_prefix}.{fmt}', format=fmt)
    plt.close(fig)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    setup_matplotlib()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  Dimensionality Reduction: DNABERT3 vs Motif Classifier FC")
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
    dropout_val = routing_info['dropout']

    classifiers = {}
    for motif_name, weight_path in routing.items():
        ckpt = torch.load(weight_path, map_location=device, weights_only=False)
        clf = MotifClassifier(
            input_dim=1536, hidden_dims=hidden_dims, dropout=dropout_val
        ).to(device)
        clf.load_state_dict(ckpt['model_state_dict'])
        clf.eval()
        classifiers[motif_name] = clf
    print(f"  Loaded {len(classifiers)} classifiers")

    print("\n[3] Extracting features from test set...")
    all_f_bert = []
    all_f_combined = []
    all_fc_vectors = []
    all_labels = []
    all_motifs = []

    for file_key, motif_name in sorted(FILE_TO_MOTIF.items()):
        test_path = os.path.join(TEST_DIR, f"test_{file_key}.csv")
        if not os.path.exists(test_path):
            print(f"  SKIP {file_key}: file not found")
            continue

        df = pd.read_csv(test_path)
        labels = df['label'].values.astype(int)
        print(f"  {file_key}: {len(df)} samples")

        dataset = FeatureExtractionDataset(df, tokenizer, seq_len=201, max_len=256)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False,
                                collate_fn=collate_fn, num_workers=0,
                                pin_memory=device.type == 'cuda')

        f_bert, f_combined = extract_features_with_bert(model, dataloader, device)
        all_f_bert.append(f_bert)
        all_f_combined.append(f_combined)

        if motif_name in classifiers:
            fc_vec = extract_fc_vector(classifiers[motif_name], f_combined, device)
        else:
            fc_vec = np.zeros((len(f_combined), 128), dtype=np.float32)
        all_fc_vectors.append(fc_vec)

        all_labels.extend(labels)
        all_motifs.extend([file_key] * len(labels))

    all_f_bert = np.concatenate(all_f_bert, axis=0)
    all_f_combined = np.concatenate(all_f_combined, axis=0)
    all_fc_vectors = np.concatenate(all_fc_vectors, axis=0)
    all_labels = np.array(all_labels)

    print(f"\n  Total: {len(all_labels)} samples")
    print(f"  F_bert: {all_f_bert.shape}")
    print(f"  FC vectors: {all_fc_vectors.shape}")

    print("\n[4] Dimensionality reduction...")

    print("  DNABERT3 + t-SNE...")
    tsne_bert = TSNE(n_components=2, random_state=SEED, perplexity=30,
                     n_iter=1000, init='pca', learning_rate='auto')
    bert_tsne_raw = tsne_bert.fit_transform(all_f_bert)

    print("  DNABERT3 + PCA...")
    pca_bert = PCA(n_components=2, random_state=SEED)
    bert_pca_raw = pca_bert.fit_transform(all_f_bert)
    print(f"    PCA explained variance: {pca_bert.explained_variance_ratio_}")

    print("  Motif Classifier FC + t-SNE...")
    tsne_fc = TSNE(n_components=2, random_state=SEED, perplexity=30,
                   n_iter=1000, init='pca', learning_rate='auto')
    fc_tsne_raw = tsne_fc.fit_transform(all_fc_vectors)

    print("  Motif Classifier FC + PCA...")
    pca_fc = PCA(n_components=2, random_state=SEED)
    fc_pca_raw = pca_fc.fit_transform(all_fc_vectors)
    print(f"    PCA explained variance: {pca_fc.explained_variance_ratio_}")

    print("\n  Min-max normalization [0, 1]...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    bert_tsne = scaler.fit_transform(bert_tsne_raw)
    bert_pca = scaler.fit_transform(bert_pca_raw)
    fc_tsne = scaler.fit_transform(fc_tsne_raw)
    fc_pca = scaler.fit_transform(fc_pca_raw)

    coords_dict = {
        'bert_tsne': bert_tsne,
        'bert_pca': bert_pca,
        'fc_tsne': fc_tsne,
        'fc_pca': fc_pca
    }

    print("\n[5] Computing clustering/classification metrics...")
    metrics_all = {}
    for key, coords in coords_dict.items():
        m = compute_metrics(coords, all_labels, seed=SEED)
        metrics_all[key] = m
        print(f"  {key}:")
        for name, val in m.items():
            print(f"    {name}: {val:.4f}")

    bert_metrics_avg = {}
    fc_metrics_avg = {}
    metric_names = list(metrics_all['bert_tsne'].keys())
    for name in metric_names:
        bert_vals = [metrics_all[k][name] for k in ['bert_tsne', 'bert_pca']]
        fc_vals = [metrics_all[k][name] for k in ['fc_tsne', 'fc_pca']]
        bert_metrics_avg[name] = np.mean(bert_vals)
        fc_metrics_avg[name] = np.mean(fc_vals)

    highlight_metric = select_best_metric(bert_metrics_avg, fc_metrics_avg)
    print(f"\n  Selected highlight metric: {highlight_metric}")
    print(f"    DNABERT3 avg: {bert_metrics_avg[highlight_metric]:.4f}")
    print(f"    Motif Classifier avg: {fc_metrics_avg[highlight_metric]:.4f}")

    print("\n[6] Plotting...")

    plot_combined_2x2(coords_dict, all_labels, metrics_all, highlight_metric,
                      os.path.join(OUTPUT_DIR, "dim_reduction_comparison"))

    for key, coords in coords_dict.items():
        save_prefix = os.path.join(OUTPUT_DIR, f"scatter_{key}")
        xlabel = 't-SNE 1' if 'tsne' in key else 'PC 1'
        ylabel = 't-SNE 2' if 'tsne' in key else 'PC 2'
        plot_single_scatter(coords, all_labels, metrics_all[key],
                            highlight_metric, xlabel, ylabel, save_prefix)

    print("\n[7] Saving data...")

    data_df = pd.DataFrame({
        'motif': all_motifs,
        'label': all_labels,
        'bert_tsne_1': bert_tsne[:, 0],
        'bert_tsne_2': bert_tsne[:, 1],
        'bert_pca_1': bert_pca[:, 0],
        'bert_pca_2': bert_pca[:, 1],
        'fc_tsne_1': fc_tsne[:, 0],
        'fc_tsne_2': fc_tsne[:, 1],
        'fc_pca_1': fc_pca[:, 0],
        'fc_pca_2': fc_pca[:, 1],
    })
    data_df.to_csv(os.path.join(OUTPUT_DIR, "dim_reduction_data.csv"), index=False)

    metrics_rows = []
    for key in coords_dict:
        row = {'feature_type': key}
        row.update(metrics_all[key])
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(OUTPUT_DIR, "metrics_all.csv"), index=False)

    summary_rows = []
    for name in metric_names:
        summary_rows.append({
            'metric': name,
            'DNABERT3_tsne': metrics_all['bert_tsne'][name],
            'DNABERT3_pca': metrics_all['bert_pca'][name],
            'DNABERT3_avg': bert_metrics_avg[name],
            'FC_tsne': metrics_all['fc_tsne'][name],
            'FC_pca': metrics_all['fc_pca'][name],
            'FC_avg': fc_metrics_avg[name],
            'improvement_avg': fc_metrics_avg[name] - bert_metrics_avg[name]
                if name not in ['Davies-Bouldin']
                else bert_metrics_avg[name] - fc_metrics_avg[name],
            'is_highlight': name == highlight_metric
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)

    pca_var_df = pd.DataFrame({
        'feature': ['DNABERT3', 'Motif Classifier FC'],
        'PC1_variance_ratio': [pca_bert.explained_variance_ratio_[0],
                               pca_fc.explained_variance_ratio_[0]],
        'PC2_variance_ratio': [pca_bert.explained_variance_ratio_[1],
                               pca_fc.explained_variance_ratio_[1]],
        'cumulative_variance': [pca_bert.explained_variance_ratio_[:2].sum(),
                                pca_fc.explained_variance_ratio_[:2].sum()]
    })
    pca_var_df.to_csv(os.path.join(OUTPUT_DIR, "pca_variance.csv"), index=False)

    print(f"\n  Results saved to: {OUTPUT_DIR}/")
    print(f"    - dim_reduction_comparison.svg/png  (2x2 combined figure)")
    print(f"    - scatter_bert_tsne.svg/png         (DNABERT3 t-SNE)")
    print(f"    - scatter_bert_pca.svg/png          (DNABERT3 PCA)")
    print(f"    - scatter_fc_tsne.svg/png           (FC t-SNE)")
    print(f"    - scatter_fc_pca.svg/png            (FC PCA)")
    print(f"    - dim_reduction_data.csv            (all coordinates + labels)")
    print(f"    - metrics_all.csv                   (all metrics per panel)")
    print(f"    - metrics_summary.csv               (comparison summary)")
    print(f"    - pca_variance.csv                  (PCA variance ratios)")
    print(f"\n  Highlight metric: {highlight_metric}")
    print(f"    DNABERT3 avg: {bert_metrics_avg[highlight_metric]:.4f}")
    print(f"    Motif Classifier avg: {fc_metrics_avg[highlight_metric]:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
