"""
m6A 甲基化预测模型 HMF — Streamlit Web 应用 (Cyber-Scientific Control Panel)
============================================================================
基于 main_general.py 的预测逻辑，提供超高颜值交互式 Web 界面。
严格不修改模型核心逻辑，仅在预处理阶段和后处理作图阶段进行扩展。
支持全局动态多档位 DPI 导出（300, 600, 900, 1200 DPI）。
"""

import os
import sys
import io
import json
import time
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import List
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fusion_models import MultimodalFusionModel

# === 常量与配置 ===
NUCLEOTIDE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4}
EIIP_DICT = {'A': 0.1260, 'C': 0.1340, 'G': 0.0806, 'T': 0.1335, 'U': 0.1335, 'N': 0.0}
DPP_DIM = 16
STRUCTURE_TO_IDX = {'.': 0, '(': 1, ')': 2, 'N': 3}
SEED = 42

# 科研常用低饱和度色谱
NATURE_COLORS = [
    '#2166AC', '#E08214', '#1B7837', '#C0392B', '#7B3294',
    '#8C510A', '#D660BD', '#525252', '#B8960C', '#3288BD',
    '#666666', '#8CBD47', '#D62728', '#0C7F8C', '#7FB5DA',
    '#E6AB02', '#4DAF4A', '#E377C2'
]
NATURE_PALETTE_TWO = ['#2166AC', '#C0392B']

# 完整的 29 维理化性质名称列表
FEATURE_NAMES_29 = (
    ['EIIP'] +
    [f'NCP_{i}' for i in range(1, 13)] +
    [f'DPP_{i}' for i in range(1, 17)]
)

_PLOTLY_FONT = 'Arial, sans-serif'
_LINE_WIDTH = 0.5

# ==========================================
# 核心模型处理逻辑 (与之前保持完全一致，不修改)
# ==========================================

def fix_all_seeds(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    unique_seqs = list(set(seqs))
    seq_to_struct = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_fold_single, unique_seqs))
    for s, struct in zip(unique_seqs, results):
        seq_to_struct[s] = struct
    return [seq_to_struct[s] for s in seqs]

def get_ncp_features(nuc: str) -> list:
    nuc = nuc.upper()
    features = [0] * 12
    if nuc == 'A': features[0], features[4], features[8] = 1, 1, 1
    elif nuc == 'C': features[1], features[5], features[9] = 1, 1, 1
    elif nuc == 'G': features[2], features[6], features[10] = 1, 1, 1
    elif nuc in ['T', 'U']: features[3], features[7], features[11] = 1, 1, 1
    return features

def calculate_physicochemical_features(seq: str) -> np.ndarray:
    seq = seq.upper()
    feature_dim = 1 + 12 + 16
    features = np.zeros((len(seq), feature_dim))
    for i, nuc in enumerate(seq):
        if i >= len(seq): break
        features[i, 0] = EIIP_DICT.get(nuc, 0.0)
        ncp_feat = get_ncp_features(nuc)
        features[i, 1:13] = ncp_feat
        prev_nuc = seq[i - 1] if i > 0 else 'N'
        dipeptide_idx = (NUCLEOTIDE_TO_IDX.get(prev_nuc, 4) * 5 + NUCLEOTIDE_TO_IDX.get(nuc, 4)) % DPP_DIM
        features[i, 13 + dipeptide_idx] = 1.0
    return features

def encode_secondary_structure(structure: str, seq_len: int = 201) -> np.ndarray:
    feature_dim = 8
    features = np.zeros((seq_len, feature_dim))
    for i, char in enumerate(structure):
        if i >= seq_len: break
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
    def __init__(self, input_dim: int = 1536, hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None: hidden_dims = [512, 128]
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)

class PredictionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, seq_len: int = 201, max_len: int = 256, text_col: str = None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_len = max_len
        if text_col is None: text_col = 'text' if 'text' in df.columns else df.columns[0]
        sequences = []
        for _, row in df.iterrows():
            seq = normalize_sequence(str(row[text_col]).strip(), seq_len)
            sequences.append(seq)
        structures = batch_predict_secondary_structure(sequences, num_workers=4)
        self.processed_data = []
        for i in range(len(sequences)):
            seq = sequences[i]
            structure = structures[i]
            phy_features = calculate_physicochemical_features(seq).astype(np.float32)
            str_features = encode_secondary_structure(structure, seq_len).astype(np.float32)
            kmer_text = " ".join([seq[j:j + 3] for j in range(len(seq) - 2)])
            encoded = self.tokenizer(kmer_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
            self.processed_data.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'phy_features': phy_features,
                'str_features': str_features,
            })

    def __len__(self): return len(self.processed_data)
    def __getitem__(self, idx):
        data = self.processed_data[idx]
        return {
            'input_ids': data['input_ids'], 'attention_mask': data['attention_mask'],
            'phy_features': torch.tensor(data['phy_features'], dtype=torch.float32),
            'str_features': torch.tensor(data['str_features'], dtype=torch.float32),
        }

def collate_fn_predict(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'phy_features': torch.stack([item['phy_features'] for item in batch]),
        'str_features': torch.stack([item['str_features'] for item in batch]),
    }

@torch.no_grad()
def extract_features_from_model(model, dataloader, device):
    model.eval()
    all_features = []
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with autocast(device_type='cuda'):
            outputs = model(batch)
        all_features.append(outputs['F_combined'].cpu().numpy())
    features = np.concatenate(all_features, axis=0)
    return features

def extract_5mer_upper_ut(seq, motif_size=5):
    seq = str(seq).strip().upper().replace('U', 'T')
    length = len(seq)
    mid_idx = length // 2
    if length >= motif_size:
        half = motif_size // 2
        return seq[mid_idx - half: mid_idx + half + 1]
    return seq

def compute_mfe_for_seq(seq: str) -> float:
    import RNA
    seq_clean = seq.upper().replace('T', 'U')
    _, mfe = RNA.fold(seq_clean)
    return mfe

def enforce_ut_conversion(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df[text_col] = df[text_col].astype(str).str.upper().str.replace('U', 'T')
    return df

@st.cache_resource
def load_models(model_dir: str, dnabert_path: str, device_str: str):
    fix_all_seeds(SEED)
    device = torch.device(device_str)
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    clf_dir = os.path.join(model_dir, "classifiers")
    trunk_path = os.path.join(ckpt_dir, "multimodal_fusion_trunk.pt")
    routing_map_path = os.path.join(clf_dir, "routing_map.json")

    ckpt = torch.load(trunk_path, map_location=device, weights_only=False)
    model_config = ckpt['config']

    model = MultimodalFusionModel(
        model_name=dnabert_path, embedding_dim=model_config['EMBEDDING_DIM'],
        phy_input_dim=model_config['PHY_FEATURE_DIM'], phy_channels=model_config['CNN_CHANNELS_PHY'],
        str_input_dim=model_config['STR_FEATURE_DIM'], str_channels=model_config['CNN_CHANNELS_STR'],
        kernel_size=model_config['KERNEL_SIZE'], recon_phy_channels=model_config['RECON_PHY_CHANNELS'],
        recon_str_channels=model_config['RECON_STR_CHANNELS'], recon_target_length=model_config['RECON_TARGET_LENGTH'],
        classifier_hidden=model_config['CLASSIFIER_HIDDEN_DIM'], dropout=model_config['DROPOUT']
    ).to(device)

    for param in model.bert_extractor.parameters(): param.requires_grad = False
    model.load_fusion_trunk_state(ckpt['fusion_trunk'])

    with open(routing_map_path, 'r') as f: routing_info = json.load(f)
    routing = routing_info['routing']
    hidden_dims = routing_info['hidden_dims']
    dropout_val = routing_info['dropout']
    default_group = routing_info.get('default_group', 'others')

    classifiers = {}
    for motif_name, weight_path in routing.items():
        weight_path_fixed = weight_path.replace('\\', '/')
        if not os.path.exists(weight_path_fixed):
            alt_path = os.path.join(clf_dir, f"classifier_{motif_name}.pt")
            if os.path.exists(alt_path): weight_path_fixed = alt_path
            else: continue
        clf_ckpt = torch.load(weight_path_fixed, map_location=device, weights_only=False)
        clf = MotifClassifier(input_dim=routing_info['input_dim'], hidden_dims=hidden_dims, dropout=dropout_val).to(device)
        clf.load_state_dict(clf_ckpt['model_state_dict'])
        clf.eval()
        classifiers[motif_name] = clf

    major_motifs = set(routing.keys())
    tokenizer = AutoTokenizer.from_pretrained(dnabert_path)
    return model, classifiers, major_motifs, default_group, tokenizer, device

def run_prediction(df: pd.DataFrame, text_col: str, model, classifiers, major_motifs, default_group, tokenizer, device, batch_size: int = 16, progress_cb=None):
    df = df.copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df['motif_5mer'] = df[text_col].apply(extract_5mer_upper_ut)
    df['motif_group'] = df['motif_5mer'].apply(lambda m: m if m in major_motifs else default_group)

    if progress_cb: progress_cb(0.1, "Extracting 1536-dim features...")
    pred_df = df[[text_col]].copy().rename(columns={text_col: 'text'})
    dataset = PredictionDataset(pred_df, tokenizer, seq_len=201, max_len=256, text_col='text')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_predict, num_workers=0, pin_memory=True)
    all_features = extract_features_from_model(model, dataloader, device)
    del dataset, dataloader; torch.cuda.empty_cache()

    if progress_cb: progress_cb(0.6, "Predicting by motif group...")
    all_probs = np.zeros(len(df))
    all_preds = np.zeros(len(df), dtype=int)

    for group_name, clf in classifiers.items():
        mask = df['motif_group'] == group_name
        indices = np.where(mask)[0]
        if len(indices) == 0: continue
        X = torch.tensor(all_features[indices], dtype=torch.float32).to(device)
        clf_probs = []
        for i in range(0, len(X), 256):
            batch_x = X[i:i + 256]
            with torch.no_grad(): p = clf(batch_x).cpu().numpy()
            clf_probs.extend(p)
        clf_probs = np.array(clf_probs)
        all_probs[indices] = clf_probs
        all_preds[indices] = (clf_probs > 0.5).astype(int)

    if progress_cb: progress_cb(0.8, "Computing MFE...")
    mfe_values = [compute_mfe_for_seq(str(seq).strip()) for seq in df[text_col]]

    result_df = df.copy()
    result_df['m6A_prob'] = all_probs
    result_df['m6A_pred'] = all_preds
    result_df['motif_group'] = df['motif_group']
    result_df['MFE'] = mfe_values

    phy_features_global, str_features_global = [], []
    for seq in df[text_col]:
        seq_norm = normalize_sequence(str(seq).strip(), 201)
        pf = calculate_physicochemical_features(seq_norm)
        phy_features_global.append(pf.mean(axis=0))
    phy_array = np.array(phy_features_global)

    for seq in df[text_col]:
        seq_norm = normalize_sequence(str(seq).strip(), 201)
        import RNA
        struct, _ = RNA.fold(seq_norm.upper().replace('T', 'U'))
        sf = encode_secondary_structure(struct, 201)
        str_features_global.append(sf.mean(axis=0))
    str_array = np.array(str_features_global)

    if progress_cb: progress_cb(0.9, "Extracting classifier hidden features...")
    bert_embeddings = all_features[:, :768]
    clf_hidden = np.zeros((len(df), 128), dtype=np.float32)
    for group_name, clf in classifiers.items():
        mask = df['motif_group'] == group_name
        indices = np.where(mask)[0]
        if len(indices) == 0: continue
        X = torch.tensor(all_features[indices], dtype=torch.float32).to(device)
        with torch.no_grad():
            for i in range(0, len(X), 256):
                batch_x = X[i:i + 256]
                hidden = clf.classifier[:-2](batch_x)
                clf_hidden[indices[i:i + 256]] = hidden.cpu().numpy()

    if progress_cb: progress_cb(1.0, "Prediction complete!")
    return result_df, all_features, bert_embeddings, phy_array, str_array, clf_hidden


# ==========================================
# 高级可视化模块 (Plotly & Matplotlib)
# ==========================================

def _apply_nature_layout(fig, width=950, height=550):
    fig.update_layout(
        font=dict(family=_PLOTLY_FONT, size=15, color='black'),
        title_font=dict(family=_PLOTLY_FONT, size=17, color='black'),
        width=width, height=height,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=70, r=40, t=50, b=70),
    )
    fig.update_xaxes(
        linewidth=_LINE_WIDTH, mirror=True, showline=True, linecolor='black',
        ticks='outside', tickwidth=_LINE_WIDTH, ticklen=4, tickcolor='black',
        title_font=dict(family=_PLOTLY_FONT, size=15),
        tickfont=dict(family=_PLOTLY_FONT, size=13),
        showgrid=False
    )
    fig.update_yaxes(
        linewidth=_LINE_WIDTH, mirror=True, showline=True, linecolor='black',
        ticks='outside', tickwidth=_LINE_WIDTH, ticklen=4, tickcolor='black',
        title_font=dict(family=_PLOTLY_FONT, size=15),
        tickfont=dict(family=_PLOTLY_FONT, size=13),
        showgrid=False
    )
    fig.update_layout(legend=dict(
        font=dict(family=_PLOTLY_FONT, size=13),
        itemsizing='constant',
        borderwidth=_LINE_WIDTH,
        bordercolor='black'
    ))
    return fig

def _setup_mpl():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.weight': 'normal',
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal',
        'font.size': 14,
        'axes.linewidth': _LINE_WIDTH,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'xtick.major.width': _LINE_WIDTH,
        'ytick.major.width': _LINE_WIDTH,
        'xtick.minor.width': _LINE_WIDTH,
        'ytick.minor.width': _LINE_WIDTH,
        'lines.linewidth': _LINE_WIDTH,
        'legend.fontsize': 14,
        'legend.frameon': True,
        'legend.edgecolor': '#333333',
        'legend.fancybox': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

# --- 图 1：基序分布帕累托图 (降序排序) ---
def fig_motif_distribution(result_df: pd.DataFrame):
    motif_counts = result_df['motif_group'].value_counts().sort_values(ascending=False)
    total = motif_counts.sum()
    labels = motif_counts.index.tolist()
    values = motif_counts.values.tolist()
    cum_percent = (motif_counts.cumsum() / total * 100).tolist()

    color_start = '#A8E6CF'
    color_end = '#84C0E9'
    cmap = mcolors.LinearSegmentedColormap.from_list('macaron', [color_start, color_end])
    bar_colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, len(labels))]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=labels, y=values, marker_color=bar_colors, width=0.65, name='Count'
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=labels, y=cum_percent, mode='lines+markers', name='Cumulative %',
        line=dict(color='grey', width=_LINE_WIDTH, dash='dash'),
        marker=dict(color='darkred', size=8)
    ), secondary_y=True)

    threshold = total * 0.015
    fig.add_hline(y=threshold, line_width=_LINE_WIDTH, line_dash="dash", line_color="red", secondary_y=False)

    fig.update_layout(title='Motif Distribution Pareto Analysis')
    fig.update_yaxes(title_text='Sequence Count', secondary_y=False)
    fig.update_yaxes(title_text='Cumulative Percentage (%)', range=[0, 105], secondary_y=True)
    return _apply_nature_layout(fig, 1050, 500)

def _save_mpl_motif_distribution(result_df, save_dir, name_base, dpi=300):
    _setup_mpl()
    motif_counts = result_df['motif_group'].value_counts().sort_values(ascending=False)
    total_count = motif_counts.sum()
    labels = motif_counts.index.tolist()
    values = motif_counts.values
    cum_percent = motif_counts.cumsum() / total_count * 100

    fig, ax1 = plt.subplots(figsize=(12, 7))
    cmap = mcolors.LinearSegmentedColormap.from_list('macaron_bg', ['#A8E6CF', '#84C0E9'])
    bar_colors = [cmap(i) for i in np.linspace(0, 1, len(labels))]

    ax1.bar(labels, values, color=bar_colors, width=0.65, linewidth=0)
    ax1.set_ylabel('Sequence Count', fontsize=20, labelpad=12)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)

    ax2 = ax1.twinx()
    ax2.plot(labels, cum_percent, color='grey', marker='o', markersize=6,
             linewidth=_LINE_WIDTH, linestyle='--', markerfacecolor='darkred', markeredgecolor='darkred')
    ax2.set_ylabel('Cumulative Percentage', fontsize=20, labelpad=12)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylim(0, 105)

    threshold_val = total_count * 0.015
    ax1.axhline(y=threshold_val, color='red', linestyle='--', linewidth=_LINE_WIDTH, zorder=5)
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{name_base}.svg"), format='svg')
    fig.savefig(os.path.join(save_dir, f"{name_base}.png"), dpi=dpi)
    plt.close(fig)

# --- 图 2：DNABERT 降维边际密度散点图 ---
def _make_joint_pca_plotly(embedding, title, result_df, annotation_col):
    scaler = StandardScaler()
    X_pca = PCA(n_components=2).fit_transform(scaler.fit_transform(embedding))
    
    if annotation_col and annotation_col in result_df.columns:
        labels = result_df[annotation_col].astype(str)
    else:
        labels = result_df['m6A_pred'].map({0: 'Non-m6A', 1: 'm6A'})
    
    df_pca = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Group': labels})
    cats = sorted(df_pca['Group'].unique())
    color_map = {cat: NATURE_COLORS[i % len(NATURE_COLORS)] for i, cat in enumerate(cats)}

    fig = px.scatter(df_pca, x="PC1", y="PC2", color="Group", color_discrete_map=color_map,
                     marginal_x="violin", marginal_y="violin", title=title, opacity=0.7)
    
    fig.update_traces(marker=dict(size=5, line=dict(width=0)), selector=dict(mode='markers'))
    fig.update_traces(line_width=_LINE_WIDTH, selector=dict(type='violin'))
    
    fig = _apply_nature_layout(fig, 850, 700)
    fig.update_layout(legend=dict(itemsizing='trace')) 
    return fig

def fig_pca_scatter(bert_embeddings, str_array, phy_array, clf_hidden, result_df, annotation_col=None):
    fig_bert = _make_joint_pca_plotly(bert_embeddings, 'DNABERT Joint PCA Mapping', result_df, annotation_col)
    fig_str = _make_joint_pca_plotly(str_array, 'RNA Secondary Structure Joint PCA Mapping', result_df, annotation_col)
    fig_phy = _make_joint_pca_plotly(phy_array, 'Physicochemical Features Joint PCA Mapping', result_df, annotation_col)
    fig_clf = _make_joint_pca_plotly(clf_hidden, 'Classifier Hidden States Joint PCA Mapping', result_df, annotation_col)
    return fig_bert, fig_str, fig_phy, fig_clf

def _save_mpl_pca_scatter(bert_embeddings, str_array, phy_array, clf_hidden, result_df, save_dir, name_base, annotation_col=None, dpi=300):
    _setup_mpl()
    embeddings_list = [
        (bert_embeddings, 'DNABERT Joint PCA Mapping', 'bert'),
        (str_array, 'RNA Secondary Structure Joint PCA Mapping', 'str'),
        (phy_array, 'Physicochemical Features Joint PCA Mapping', 'phy'),
        (clf_hidden, 'Classifier Hidden States Joint PCA Mapping', 'clf'),
    ]

    if annotation_col and annotation_col in result_df.columns:
        labels = result_df[annotation_col].astype(str).values
    else:
        labels = result_df['m6A_pred'].map({0: 'Non-m6A', 1: 'm6A'}).values
        
    cats = sorted(set(labels))
    palette = {cat: NATURE_COLORS[i % len(NATURE_COLORS)] for i, cat in enumerate(cats)}

    for emb, title, suffix in embeddings_list:
        X_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(emb))
        df_pca = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Group': labels})
        
        g = sns.JointGrid(data=df_pca, x="PC1", y="PC2", hue="Group", palette=palette, height=8)
        g.plot_joint(sns.scatterplot, s=25, alpha=0.7, edgecolor='none', linewidth=0)
        g.plot_marginals(sns.kdeplot, fill=True, alpha=0.4, linewidth=_LINE_WIDTH)
        
        g.ax_joint.set_title(title, pad=20, fontsize=16)
        g.ax_joint.legend(markerscale=2.0, edgecolor='black', frameon=True)
        for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:
            for spine in ax.spines.values(): spine.set_linewidth(_LINE_WIDTH)
            ax.tick_params(width=_LINE_WIDTH)
            
        g.savefig(os.path.join(save_dir, f"{name_base}_{suffix}.svg"))
        g.savefig(os.path.join(save_dir, f"{name_base}_{suffix}.png"), dpi=dpi)
        plt.close(g.fig)

# --- 显著性检验工具函数 ---
def _auto_significance_test(group1, group2):
    from scipy import stats as sp_stats
    n1, n2 = len(group1), len(group2)
    if n1 < 3 or n2 < 3:
        return 'N/A', 1.0, 'Insufficient samples'
    _, norm_p1 = sp_stats.shapiro(group1[:5000])
    _, norm_p2 = sp_stats.shapiro(group2[:5000])
    is_normal = (norm_p1 > 0.05) and (norm_p2 > 0.05)
    if is_normal:
        _, lev_p = sp_stats.levene(group1, group2)
        equal_var = lev_p > 0.05
        if equal_var:
            stat, pval = sp_stats.ttest_ind(group1, group2, equal_var=True)
            method = "Student's t-test"
        else:
            stat, pval = sp_stats.ttest_ind(group1, group2, equal_var=False)
            method = "Welch's t-test"
    else:
        stat, pval = sp_stats.mannwhitneyu(group1, group2, alternative='two-sided')
        method = "Mann-Whitney U"
    if pval < 0.001:
        sig_symbol = '***'
    elif pval < 0.01:
        sig_symbol = '**'
    elif pval < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'
    return sig_symbol, pval, method

def _sig_annotation(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'

# --- 图 3a: MFE 分布图 ---
def fig_mfe_violin(result_df: pd.DataFrame):
    m6a_mfe = result_df.loc[result_df['m6A_pred'] == 1, 'MFE'].values
    non_m6a_mfe = result_df.loc[result_df['m6A_pred'] == 0, 'MFE'].values

    sig_symbol, pval, method = _auto_significance_test(non_m6a_mfe, m6a_mfe)
    pval_str = f'p={pval:.2e}' if pval < 0.001 else f'p={pval:.4f}'
    sig_text = f'{sig_symbol} ({method}, {pval_str})'

    fig = go.Figure()
    fig.add_trace(go.Violin(
        y=non_m6a_mfe, name='Non-m6A', box_visible=True, meanline_visible=True,
        line_color=NATURE_PALETTE_TWO[0], fillcolor=NATURE_PALETTE_TWO[0], opacity=0.65,
        spanmode='soft', points=False, line_width=_LINE_WIDTH, box_line_width=_LINE_WIDTH,
    ))
    fig.add_trace(go.Violin(
        y=m6a_mfe, name='m6A', box_visible=True, meanline_visible=True,
        line_color=NATURE_PALETTE_TWO[1], fillcolor=NATURE_PALETTE_TWO[1], opacity=0.65,
        spanmode='soft', points=False, line_width=_LINE_WIDTH, box_line_width=_LINE_WIDTH,
    ))

    y_max = max(np.max(non_m6a_mfe), np.max(m6a_mfe)) if len(non_m6a_mfe) > 0 and len(m6a_mfe) > 0 else 0
    y_min = min(np.min(non_m6a_mfe), np.min(m6a_mfe)) if len(non_m6a_mfe) > 0 and len(m6a_mfe) > 0 else 0
    y_range = y_max - y_min if y_max != y_min else 1
    bracket_y = y_max + y_range * 0.05
    text_y = y_max + y_range * 0.12

    fig.add_shape(type='line', x0=0, x1=0, y0=bracket_y, y1=bracket_y + y_range * 0.03, xref='x', yref='y',
                  line=dict(color='black', width=1))
    fig.add_shape(type='line', x0=1, x1=1, y0=bracket_y, y1=bracket_y + y_range * 0.03, xref='x', yref='y',
                  line=dict(color='black', width=1))
    fig.add_shape(type='line', x0=0, x1=1, y0=bracket_y + y_range * 0.03, y1=bracket_y + y_range * 0.03, xref='x', yref='y',
                  line=dict(color='black', width=1))
    fig.add_annotation(x=0.5, y=text_y, text=sig_text, showarrow=False,
                       font=dict(size=13, family='Arial, sans-serif'), xref='x', yref='y')

    fig.update_layout(title='Minimum Free Energy (MFE) Distribution Profile', yaxis_title='MFE (kcal/mol)')
    return _apply_nature_layout(fig, 650, 550)

def _save_mpl_mfe_violin(result_df, save_dir, name_base, dpi=300):
    _setup_mpl()
    m6a_mfe = result_df.loc[result_df['m6A_pred'] == 1, 'MFE'].values
    non_m6a_mfe = result_df.loc[result_df['m6A_pred'] == 0, 'MFE'].values

    sig_symbol, pval, method = _auto_significance_test(non_m6a_mfe, m6a_mfe)
    pval_str = f'p={pval:.2e}' if pval < 0.001 else f'p={pval:.4f}'
    sig_text = f'{sig_symbol} ({method}, {pval_str})'

    fig, ax = plt.subplots(figsize=(6, 5.5))
    parts = ax.violinplot([non_m6a_mfe, m6a_mfe], positions=[1, 2], showmeans=True, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(NATURE_PALETTE_TWO[i])
        pc.set_alpha(0.65)
        pc.set_edgecolor('black')
        pc.set_linewidth(_LINE_WIDTH)
    for key in ['cmeans', 'cmedians']:
        parts[key].set_linewidth(_LINE_WIDTH)
        parts[key].set_color('black')

    y_max = max(np.max(non_m6a_mfe), np.max(m6a_mfe))
    y_range = y_max - min(np.min(non_m6a_mfe), np.min(m6a_mfe)) if y_max != min(np.min(non_m6a_mfe), np.min(m6a_mfe)) else 1
    bracket_y = y_max + y_range * 0.05
    ax.plot([1, 1, 2, 2], [bracket_y, bracket_y + y_range * 0.03, bracket_y + y_range * 0.03, bracket_y],
            color='black', linewidth=1)
    ax.text(1.5, bracket_y + y_range * 0.06, sig_text, ha='center', va='bottom', fontsize=11)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Non-m6A', 'm6A'])
    ax.set_ylabel('MFE (kcal/mol)')
    ax.set_title('Minimum Free Energy (MFE) Distribution')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(save_dir, f"{name_base}.svg"), format='svg')
    fig.savefig(os.path.join(save_dir, f"{name_base}.png"), dpi=dpi)
    plt.close(fig)

# --- 图 3b: 29 维度理化性质原始箱线图 ---
def fig_phy_boxplot(phy_array: np.ndarray, result_df: pd.DataFrame):
    m6a_mask = result_df['m6A_pred'].values == 1
    non_m6a_mask = result_df['m6A_pred'].values == 0

    fig = go.Figure()
    sig_annotations = []
    for idx, name in enumerate(FEATURE_NAMES_29):
        y_non = phy_array[non_m6a_mask, idx]
        y_m6a = phy_array[m6a_mask, idx]
        
        fig.add_trace(go.Box(
            y=y_non, x=[name] * len(y_non), name='Non-m6A', legendgroup='Non-m6A',
            marker_color=NATURE_PALETTE_TWO[0], line_width=_LINE_WIDTH, boxpoints=False, opacity=0.75,
            showlegend=True if idx == 0 else False
        ))
        
        fig.add_trace(go.Box(
            y=y_m6a, x=[name] * len(y_m6a), name='m6A', legendgroup='m6A',
            marker_color=NATURE_PALETTE_TWO[1], line_width=_LINE_WIDTH, boxpoints=False, opacity=0.75,
            showlegend=True if idx == 0 else False
        ))

        sig_symbol, pval, method = _auto_significance_test(y_non, y_m6a)
        sig_annotations.append(dict(
            x=name, y=max(np.max(y_non), np.max(y_m6a)) if len(y_non) > 0 and len(y_m6a) > 0 else 0,
            text=sig_symbol, showarrow=False, font=dict(size=10, family='Arial, sans-serif'),
            yshift=8
        ))

    fig.update_layout(
        title='Complete Physicochemical Profile Across All 29 Dimensions',
        yaxis_title='Normalized Feature Value', boxmode='group', boxgap=0.2, boxgroupgap=0.05,
        annotations=sig_annotations
    )

    first_sig = next((a for a in sig_annotations if a['text'] != 'ns'), None)
    if first_sig:
        method_text = f'Significance: {method} (* p<0.05, ** p<0.01, *** p<0.001, ns: not significant)'
        fig.add_annotation(xref='paper', yref='paper', x=0.5, y=-0.22, text=method_text,
                           showarrow=False, font=dict(size=10, family='Arial, sans-serif'))

    return _apply_nature_layout(fig, 1400, 600)

def _save_mpl_phy_boxplot(phy_array, result_df, save_dir, name_base, dpi=300):
    _setup_mpl()
    m6a_mask = result_df['m6A_pred'].values == 1
    non_m6a_mask = result_df['m6A_pred'].values == 0

    fig, ax = plt.subplots(figsize=(16, 7))
    data_non = [phy_array[non_m6a_mask, i] for i in range(29)]
    data_m6a = [phy_array[m6a_mask, i] for i in range(29)]

    positions_non = [i * 2.5 for i in range(29)]
    positions_m6a = [i * 2.5 + 0.8 for i in range(29)]

    bp1 = ax.boxplot(data_non, positions=positions_non, widths=0.6, patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor=NATURE_PALETTE_TWO[0], alpha=0.75, linewidth=_LINE_WIDTH, edgecolor='black'),
                     medianprops=dict(color='black', linewidth=_LINE_WIDTH),
                     whiskerprops=dict(linewidth=_LINE_WIDTH, color='black'),
                     capprops=dict(linewidth=_LINE_WIDTH, color='black'))

    bp2 = ax.boxplot(data_m6a, positions=positions_m6a, widths=0.6, patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor=NATURE_PALETTE_TWO[1], alpha=0.75, linewidth=_LINE_WIDTH, edgecolor='black'),
                     medianprops=dict(color='black', linewidth=_LINE_WIDTH),
                     whiskerprops=dict(linewidth=_LINE_WIDTH, color='black'),
                     capprops=dict(linewidth=_LINE_WIDTH, color='black'))

    used_method = None
    for i in range(29):
        sig_symbol, pval, method = _auto_significance_test(data_non[i], data_m6a[i])
        if used_method is None and sig_symbol != 'ns':
            used_method = method
        x_pos = i * 2.5 + 0.4
        y_pos = max(np.max(data_non[i]), np.max(data_m6a[i])) if len(data_non[i]) > 0 and len(data_m6a[i]) > 0 else 0
        ax.text(x_pos, y_pos, sig_symbol, ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks([i * 2.5 + 0.4 for i in range(29)])
    ax.set_xticklabels(FEATURE_NAMES_29, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Feature Value')
    ax.set_title('Complete Physicochemical Profile Across All 29 Dimensions')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Non-m6A', 'm6A'], frameon=True, edgecolor='black')
    if used_method:
        ax.text(0.5, -0.15, f'Significance: {used_method} (* p<0.05, ** p<0.01, *** p<0.001, ns: not significant)',
                ha='center', transform=ax.transAxes, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{name_base}.svg"), format='svg')
    fig.savefig(os.path.join(save_dir, f"{name_base}.png"), dpi=dpi)
    plt.close(fig)

# --- 图 5: 基序 & 标签双向交叉聚类丰度热图 ---
def _save_mpl_clustermap(result_df, save_dir, name_base, annotation_col=None, dpi=300):
    _setup_mpl()
    group_col = annotation_col if annotation_col and annotation_col in result_df.columns else 'm6A_pred'
    cross_tab = pd.crosstab(result_df['motif_group'], result_df[group_col])
    if cross_tab.empty or cross_tab.shape[0] < 2 or cross_tab.shape[1] < 2: return 

    data_norm = cross_tab.apply(lambda x: (x - x.mean()) / (x.std() + 1e-9), axis=1)
    g = sns.clustermap(
        data_norm, cmap='RdYlBu_r', linewidths=0, linecolor=None,
        figsize=(10, 8), tree_kws={'linewidths': _LINE_WIDTH}, dendrogram_ratio=(0.15, 0.15)
    )
    ax = g.ax_heatmap
    ax.set_xlabel('Sample Group Labels' if group_col == 'm6A_pred' else group_col, fontsize=16, labelpad=10)
    ax.set_ylabel('Motif Group Categories', fontsize=16, labelpad=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=13)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=13)
    g.ax_cbar.set_title('Z-score', fontsize=13)
    g.ax_cbar.tick_params(labelsize=11, width=_LINE_WIDTH)
    
    g.savefig(os.path.join(save_dir, f"{name_base}.svg"))
    g.savefig(os.path.join(save_dir, f"{name_base}.png"), dpi=dpi)
    plt.close(g.fig)

def fig_clustermap_plotly(result_df, annotation_col=None):
    group_col = annotation_col if annotation_col and annotation_col in result_df.columns else 'm6A_pred'
    cross_tab = pd.crosstab(result_df['motif_group'], result_df[group_col])
    data_norm = cross_tab.apply(lambda x: (x - x.mean()) / (x.std() + 1e-9), axis=1)
    
    fig = px.imshow(data_norm, color_continuous_scale='RdYlBu_r', aspect="auto",
                    labels=dict(x="Group Labels", y="Motif Categories", color="Z-score"))
    fig.update_layout(title="Motif-Group Enrichment Heatmap Profiling")
    return _apply_nature_layout(fig, 850, 600)

# --- 图 6: 基序-29维理化性质全维度关联热图 ---
def fig_motif_phy_heatmap_plotly(phy_array, result_df):
    motifs = sorted(result_df['motif_group'].unique())
    matrix = np.zeros((len(motifs), 29))
    for i, m in enumerate(motifs):
        mask = result_df['motif_group'] == m
        if mask.sum() > 0: matrix[i] = phy_array[mask].mean(axis=0)
            
    df_matrix = pd.DataFrame(matrix, index=motifs, columns=FEATURE_NAMES_29)
    df_norm = df_matrix.apply(lambda x: (x - x.mean()) / (x.std() + 1e-9), axis=0)

    fig = px.imshow(df_norm, color_continuous_scale='Viridis', aspect="auto",
                    labels=dict(x="Physicochemical Features", y="Motif Group", color="Z-score"))
    fig.update_layout(title="Motif Categories vs 29-Dimensional Physicochemical Properties Heatmap")
    return _apply_nature_layout(fig, 1200, 600)

def _save_mpl_motif_phy_heatmap(phy_array, result_df, save_dir, name_base, dpi=300):
    _setup_mpl()
    motifs = sorted(result_df['motif_group'].unique())
    matrix = np.zeros((len(motifs), 29))
    for i, m in enumerate(motifs):
        mask = result_df['motif_group'] == m
        if mask.sum() > 0: matrix[i] = phy_array[mask].mean(axis=0)
        
    df_matrix = pd.DataFrame(matrix, index=motifs, columns=FEATURE_NAMES_29)
    df_norm = df_matrix.apply(lambda x: (x - x.mean()) / (x.std() + 1e-9), axis=0)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(df_norm, cmap='viridis', cbar_kws={'label': 'Z-score'}, linewidths=0, ax=ax)
    ax.set_xlabel('Physicochemical Features (29 Dimensions)', fontsize=15, labelpad=10)
    ax.set_ylabel('Motif Group Categories', fontsize=15, labelpad=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{name_base}.svg"), format='svg')
    fig.savefig(os.path.join(save_dir, f"{name_base}.png"), dpi=dpi)
    plt.close(fig)

def save_fig_to_file(fig, save_dir: str, name_base: str, mpl_func=None, mpl_args=None, dpi=300):
    os.makedirs(save_dir, exist_ok=True)
    html_path = os.path.join(save_dir, f"{name_base}.html")
    try: fig.write_html(html_path, include_plotlyjs='cdn', full_html=True)
    except: pass
    if mpl_func is not None and mpl_args is not None:
        try: mpl_func(*mpl_args, save_dir, name_base, dpi=dpi)
        except: pass

# ==========================================
# UI 渲染层 (Streamlit)
# ==========================================
def main():
    st.set_page_config(page_title="HMF-m6A", page_icon="🧬", layout="wide")

    # CSS 强力注入控制台赛博美学
    st.markdown("""
    <style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
    * { font-family: 'Arial', sans-serif !important; font-weight: normal !important; }
    
    section[data-testid="stSidebar"] {
        background-color: #fcfdfe;
        border-right: 0.5px solid #e0e6ed;
    }
    button[kind="header"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        padding: 0 !important;
        border-radius: 6px !important;
        background-color: #f0f4f8 !important;
        border: 0.5px solid #d3dfe9 !important;
        color: #4C72B0 !important;
        font-size: 0 !important;
        line-height: 0 !important;
        overflow: hidden !important;
    }
    button[kind="header"]::before {
        content: "☰" !important;
        font-size: 16px !important;
        line-height: 32px !important;
    }
    button[kind="header"]:hover {
        background-color: #4C72B0 !important;
        color: white !important;
    }
    .title-container { 
        display: flex; align-items: center; justify-content: center; 
        flex-direction: column; margin-bottom: 2.5rem; margin-top: 1rem;
        background: #fafbfc;
        border: 0.5px solid #e2e8f0;
        border-radius: 8px;
        padding: 22px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.015);
    }
    .spin-icon { 
        animation: spin-kf 10s linear infinite; 
        display: inline-block; font-size: 3.5rem; margin-bottom: 0.5rem;
    }
    @keyframes spin-kf { 100% { transform: rotate(360deg); } }
    
    .main-title { 
        font-size: 3.2rem; 
        margin: 0; 
        letter-spacing: 2px;
        background: linear-gradient(90deg, #3A6073, #3a7bd5, #00d2ff, #3A6073);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: flow-text-kf 6s linear infinite;
        text-shadow: 0 0 15px rgba(0, 210, 255, 0.05);
    }
    @keyframes flow-text-kf { to { background-position: 200% center; } }
    .sub-title { font-size: 1.25rem; color: #788896; margin-top: 0.6rem; letter-spacing: 0.5px; }
    
    button[data-testid="stBaseButton-tab"] {
        background-color: #f8fafc !important;
        color: #5c6b73 !important;
        border: 0.5px solid #d3dfe9 !important;
        border-radius: 4px !important;
        padding: 6px 18px !important;
        margin-right: 4px !important;
    }
    button[data-testid="stBaseButton-tab"][aria-selected="true"] {
        background-color: #4C72B0 !important;
        color: white !important;
        border: 0.5px solid #3b5984 !important;
    }
    div[data-testid="stButton"] button {
        background-color: #f8fafc !important;
        color: #4C72B0 !important;
        border: 0.5px solid #4C72B0 !important;
        border-radius: 4px !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #4C72B0 !important;
        color: white !important;
    }
    .history-card {
        border: 0.5px solid #cbd5e1; 
        border-left: 3px solid #4C72B0; 
        padding: 10px; 
        border-radius: 4px; 
        margin-bottom: 6px; 
        background-color: #f8fafc;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="title-container">
        <div class="spin-icon">🧬</div>
        <h1 class="main-title">HMF-m6A Web UI</h1>
        <div class="sub-title">Hierarchical Multimodal m6A RNA Methylation Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    if 'prediction_history' not in st.session_state: st.session_state.prediction_history = []
    if 'current_view' not in st.session_state: st.session_state.current_view = 'input'
    if 'active_result_key' not in st.session_state: st.session_state.active_result_key = None
    if 'uploaded_file_data' not in st.session_state: st.session_state.uploaded_file_data = None
    if 'uploaded_file_name' not in st.session_state: st.session_state.uploaded_file_name = None
    if 'annotation_file_data' not in st.session_state: st.session_state.annotation_file_data = None
    if 'annotation_file_name' not in st.session_state: st.session_state.annotation_file_name = None

    if st.session_state.current_view == 'results':
        _render_results_page()
        return

    with st.sidebar:
        st.header("⚙️ Configuration")
        model_dir = st.text_input("Model Root Directory", value=PROJECT_ROOT)
        dnabert_path = st.text_input("DNABERT3 Path", value=os.path.join(PROJECT_ROOT, "DNABERT3"))
        output_dir = st.text_input("Output Directory", value=os.path.join(PROJECT_ROOT, "web_results"))
        batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=16)

        st.markdown("---")
        st.header("📁 Input Data")
        uploaded_file = st.file_uploader("Upload CSV/TSV File", type=['csv', 'tsv'])
        annotation_file = st.file_uploader("Upload Annotation CSV (Optional)", type=['csv', 'tsv'])
        st.markdown("---")
        st.markdown("**Note**: All sequences are force-converted from U to T before DNABERT input.")

    if uploaded_file is not None:
        st.session_state.uploaded_file_data = uploaded_file.getvalue()
        st.session_state.uploaded_file_name = uploaded_file.name
    if annotation_file is not None:
        st.session_state.annotation_file_data = annotation_file.getvalue()
        st.session_state.annotation_file_name = annotation_file.name

    if st.session_state.prediction_history:
        st.subheader("📋 Console History Ledgers")
        for i, record in enumerate(reversed(st.session_state.prediction_history)):
            idx = len(st.session_state.prediction_history) - 1 - i
            st.markdown(f"""
            <div class="history-card">
                <div style="font-size: 0.95rem; color: #1e293b; margin-bottom: 4px;">📂 {record['name']}</div>
                <div style="font-size: 0.85rem; color: #64748b;">🧾 {record['n_samples']} items | Cost: {record['elapsed']:.1f}s</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Dashboard ►", key=f"view_hist_{idx}", use_container_width=True):
                st.session_state.active_result_key = idx
                st.session_state.current_view = 'results'
                st.rerun()
        st.markdown("---")

    if not torch.cuda.is_available():
        st.error("⚠️ GPU/CUDA not detected. This application requires GPU!")
        st.stop()

    if uploaded_file is None and st.session_state.uploaded_file_data is None:
        st.info("👈 Please upload a CSV file containing RNA sequences to start prediction.")
        st.markdown("""
        ### Input File Requirements
        - CSV or TSV format
        - Must contain a column with RNA sequences (201 nt context)
        - Sequences may contain U or T; U will be auto-converted to T

        ### Optional: Annotation File
        - CSV with group/annotation columns
        - Used for coloring in PCA scatter plot
        """)
        st.stop()

    if uploaded_file is not None:
        file_data = uploaded_file
        file_name = uploaded_file.name
    else:
        file_data = io.BytesIO(st.session_state.uploaded_file_data)
        file_name = st.session_state.uploaded_file_name

    df = pd.read_csv(file_data, sep='\t' if file_name.endswith('.tsv') else ',')
    st.subheader("Data Preview Matrix")
    st.dataframe(df.head(10), use_container_width=True)
    
    text_col = st.selectbox("Select Sequence Column", options=df.columns.tolist(), index=0 if 'text' not in df.columns else df.columns.tolist().index('text'))

    annotation_col = None
    annotation_df = None
    if annotation_file is not None or st.session_state.annotation_file_data is not None:
        if annotation_file is not None:
            anno_data = annotation_file
            anno_name = annotation_file.name
        else:
            anno_data = io.BytesIO(st.session_state.annotation_file_data)
            anno_name = st.session_state.annotation_file_name
        annotation_df = pd.read_csv(anno_data, sep='\t' if anno_name.endswith('.tsv') else ',')
        annotation_df = annotation_df.loc[:, ~annotation_df.columns.duplicated()]
        st.subheader("Annotation Preview Matrix")
        st.dataframe(annotation_df.head(5), use_container_width=True)
        annotation_col = st.selectbox("Select Annotation/Group Column", options=annotation_df.columns.tolist())

    if st.button("🚀 Run Multimodal Inference Architecture", type="primary", use_container_width=True):
        df = enforce_ut_conversion(df, text_col)
        with st.spinner("Loading state dict tokens into GPU tensors..."):
            model, classifiers, major_motifs, default_group, tokenizer, device = load_models(model_dir, dnabert_path, 'cuda')

        if annotation_df is not None and annotation_col is not None:
            if annotation_col in df.columns:
                df[annotation_col] = df[annotation_col].astype(str)
            elif len(annotation_df) == len(df):
                df[annotation_col] = annotation_df[annotation_col].astype(str).values
            else:
                merge_col = st.selectbox("Select Merge Key Column (Main File)", options=df.columns.tolist())
                merge_col_anno = st.selectbox("Select Merge Key Column (Annotation File)", options=annotation_df.columns.tolist())
                df[merge_col] = df[merge_col].astype(str)
                annotation_df[merge_col_anno] = annotation_df[merge_col_anno].astype(str)
                if merge_col_anno == annotation_col:
                    df = df.merge(annotation_df[[annotation_col]], left_on=merge_col, right_on=annotation_col, how='left')
                else:
                    df = df.merge(annotation_df[[merge_col_anno, annotation_col]], left_on=merge_col, right_on=merge_col_anno, how='left')
                    df.drop(columns=[merge_col_anno], inplace=True, errors='ignore')
                df[annotation_col] = df[annotation_col].astype(str)

        progress_bar = st.progress(0, text="Initializing matrix buffers...")
        t0 = time.time()
        result_df, all_features, bert_embeddings, phy_array, str_array, clf_hidden = run_prediction(
            df, text_col, model, classifiers, major_motifs, default_group, tokenizer, device,
            batch_size=batch_size, progress_cb=lambda f, m: progress_bar.progress(f, text=m)
        )
        elapsed = time.time() - t0

        os.makedirs(output_dir, exist_ok=True)
        result_df.to_csv(os.path.join(output_dir, "prediction_results.csv"), index=False)

        st.session_state.prediction_history.append({
            'name': file_name, 'n_samples': len(result_df), 'elapsed': elapsed,
            'result_df': result_df, 'bert_embeddings': bert_embeddings, 'phy_array': phy_array,
            'str_array': str_array, 'clf_hidden': clf_hidden, 'text_col': text_col,
            'annotation_col': annotation_col, 'output_dir': output_dir,
        })
        st.session_state.active_result_key = len(st.session_state.prediction_history) - 1
        st.session_state.current_view = 'results'
        st.rerun()

def _render_results_page():
    idx = st.session_state.active_result_key
    record = st.session_state.prediction_history[idx]
    result_df, bert_embeddings, phy_array, str_array, clf_hidden = record['result_df'], record['bert_embeddings'], record['phy_array'], record['str_array'], record['clf_hidden']
    text_col, annotation_col, output_dir = record['text_col'], record['annotation_col'], record['output_dir']

    col_back, col_title = st.columns([1, 10])
    with col_back:
        if st.button("⬅ Back"):
            st.session_state.current_view = 'input'
            st.session_state.uploaded_file_data = None
            st.session_state.uploaded_file_name = None
            st.session_state.annotation_file_data = None
            st.session_state.annotation_file_name = None
            st.rerun()
            
    with col_title:
        st.subheader(f"📊 Analytical Dashboard — {record['name']}")
    st.caption(f"{record['n_samples']} items loaded | Computational Cost: {record['elapsed']:.1f}s")

    display_cols = [c for c in [text_col, 'm6A_prob', 'm6A_pred', 'motif_group', 'MFE', annotation_col] if c and c in result_df.columns]
    st.dataframe(result_df[display_cols], use_container_width=True)

    st.download_button("📥 Export Matrix Tabular Database (.csv)", data=result_df.to_csv(index=False).encode('utf-8'), file_name="prediction_results.csv", mime="text/csv")
    
    st.markdown("---")
    
    # 【新增点】高颜值 DPI 动态选择器栏
    c_space, c_dpi = st.columns([3, 1])
    with c_dpi:
        dpi_choice = st.selectbox("🎯 Export Figures Resolution (DPI)", [300, 600, 900, 1200], index=0, help="Dynamically adjust resolution parameter for exported PNG figures.")

    st.subheader("📈 Computational Bio-Visualization Decks")
    tab1, tab2, tab3, tab4 = st.tabs(["1. Motif Distribution Analysis", "2. High-Dim Space PCA Reduction", "3. Cross-talk Enrichment Clustermap", "4. Multi-modal Structural Profiles"])

    # === Tab 1: 基序分布 ===
    with tab1:
        st.markdown("#### Fig 1: Motif Group Distribution (Sorted Descending Pareto Chart)")
        fig1 = fig_motif_distribution(result_df)
        st.plotly_chart(fig1, use_container_width=True)
        # 将全局选择的 DPI 传入保存函数中
        save_fig_to_file(fig1, output_dir, "fig1_motif_distribution", _save_mpl_motif_distribution, (result_df,), dpi=dpi_choice)
        _dl_btns(output_dir, "fig1_motif_distribution", f"fig1_{idx}")

    # === Tab 2: 降维空间图（彻底解耦，每张图均配备独立按钮） ===
    with tab2:
        st.markdown("#### Fig 2: Latent Embeddings Joint Mapping with Marginal Density Projections")
        fig2_bert, fig2_str, fig2_phy, fig2_clf = fig_pca_scatter(bert_embeddings, str_array, phy_array, clf_hidden, result_df, annotation_col)
        
        # 触发全套图片的后台高级 Matplotlib 引擎渲染保存
        try:
            _save_mpl_pca_scatter(bert_embeddings, str_array, phy_array, clf_hidden, result_df, output_dir, "fig2_pca", annotation_col, dpi=dpi_choice)
        except Exception:
            pass
        
        # 逐张图表完美平铺展现并分配精准的下载按钮
        st.markdown("**Fig 2a: DNABERT Sequence Space**")
        st.plotly_chart(fig2_bert, use_container_width=True)
        _dl_btns(output_dir, "fig2_pca_bert", f"pca_bert_{idx}")
        
        st.markdown("**Fig 2b: Secondary Structure Topology**")
        st.plotly_chart(fig2_str, use_container_width=True)
        _dl_btns(output_dir, "fig2_pca_str", f"pca_str_{idx}")
        
        st.markdown("**Fig 2c: 29D Physicochemical State**")
        st.plotly_chart(fig2_phy, use_container_width=True)
        _dl_btns(output_dir, "fig2_pca_phy", f"pca_phy_{idx}")
        
        st.markdown("**Fig 2d: MoE Deep Hidden States**")
        st.plotly_chart(fig2_clf, use_container_width=True)
        _dl_btns(output_dir, "fig2_pca_clf", f"pca_clf_{idx}")

    # === Tab 3: 交叉热图 ===
    with tab3:
        st.markdown("#### Fig 3: Motif vs Sample Group Dual Hierarchical Clustermap Matrix")
        fig_hm = fig_clustermap_plotly(result_df, annotation_col)
        st.plotly_chart(fig_hm, use_container_width=True)
        save_fig_to_file(fig_hm, output_dir, "fig3_motif_clustermap", dpi=dpi_choice)
        try:
            _save_mpl_clustermap(result_df, output_dir, "fig3_motif_clustermap", annotation_col=annotation_col, dpi=dpi_choice)
        except Exception:
            pass
        _dl_btns(output_dir, "fig3_motif_clustermap", f"fig3_{idx}")

    # === Tab 4: 理化性质与二级结构结构特征面板 ===
    with tab4:
        st.markdown("#### Fig 4a: RNA Thermodynamic Secondary Structural Stability (MFE Distribution Profile)")
        fig4a = fig_mfe_violin(result_df)
        st.plotly_chart(fig4a, use_container_width=True)
        save_fig_to_file(fig4a, output_dir, "fig4a_mfe_violin", _save_mpl_mfe_violin, (result_df,), dpi=dpi_choice)
        _dl_btns(output_dir, "fig4a_mfe_violin", f"fig4a_{idx}")

        st.markdown("#### Fig 4b: Panoramic View Across Complete 29-Dimensional Physicochemical Coordinates")
        fig4b = fig_phy_boxplot(phy_array, result_df)
        st.plotly_chart(fig4b, use_container_width=True)
        save_fig_to_file(fig4b, output_dir, "fig4b_phy_boxplot", _save_mpl_phy_boxplot, (phy_array, result_df), dpi=dpi_choice)
        _dl_btns(output_dir, "fig4b_phy_boxplot", f"fig4b_{idx}")

        st.markdown("#### Fig 4c: Sequence Motif Group vs All 29-Dimensional Biophysical Fingerprint Cross-Heatmap")
        fig4c = fig_motif_phy_heatmap_plotly(phy_array, result_df)
        st.plotly_chart(fig4c, use_container_width=True)
        save_fig_to_file(fig4c, output_dir, "fig4c_motif_phy_heatmap", dpi=dpi_choice)
        try:
            _save_mpl_motif_phy_heatmap(phy_array, result_df, output_dir, "fig4c_motif_phy_heatmap", dpi=dpi_choice)
        except Exception:
            pass
        _dl_btns(output_dir, "fig4c_motif_phy_heatmap", f"fig4c_{idx}")

def _dl_btns(out_dir, base, unique_key):
    """通用极简科研风格多格式下载网格按钮组"""
    c1, c2 = st.columns(2)
    with c1:
        p_svg = os.path.join(out_dir, f"{base}.svg")
        if os.path.exists(p_svg):
            with open(p_svg, "rb") as f: 
                st.download_button("📥 Download Vector SVG", f, file_name=f"{base}.svg", mime="image/svg+xml", key=f"svg_{base}_{unique_key}")
    with c2:
        p_png = os.path.join(out_dir, f"{base}.png")
        if os.path.exists(p_png):
            with open(p_png, "rb") as f: 
                st.download_button("📥 Download Publication PNG", f, file_name=f"{base}.png", mime="image/png", key=f"png_{base}_{unique_key}")

if __name__ == "__main__":
    main()