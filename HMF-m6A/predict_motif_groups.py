"""
对 data/human_hg38_m6A_result_col29_with_mapped_loc_final.csv 进行:
1. chr_context 列大写 + U->T, 提取中心5-mer motif
2. 按照原有 motif classifier 分组 (major_motifs -> 对应组, 其余 -> other)
3. 保存分组后的文件
4. 分层抽样 ~10000 条, 各组尽量均衡
5. 用训练好的融合模型提取1536维特征 + motif classifier 预测
6. 保存每个组各序列的预测结果
7. 输出正样本比例 (整体 + 各组)
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
DATA_PATH = os.path.join(BASE_DIR, "data", "human_hg38_m6A_result_col29_with_mapped_loc_final.csv")
GROUP_INFO_PATH = os.path.join(BASE_DIR, "grouped_data", "group_info.json")
ROUTING_MAP_PATH = os.path.join(BASE_DIR, "classifiers", "routing_map.json")
TRUNK_PATH = os.path.join(BASE_DIR, "checkpoints", "multimodal_fusion_trunk.pt")
DNABERT3_PATH = os.path.join(BASE_DIR, "DNABERT3")

OUTPUT_DIR = os.path.join(BASE_DIR, "motif_predict_results")
GROUPED_CSV_PATH = os.path.join(OUTPUT_DIR, "grouped_data.csv")
SAMPLED_CSV_PATH = os.path.join(OUTPUT_DIR, "sampled_data.csv")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
POSITIVE_RATIO_PATH = os.path.join(OUTPUT_DIR, "positive_ratio_summary.csv")

NUCLEOTIDE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4}
EIIP_DICT = {'A': 0.1260, 'C': 0.1340, 'G': 0.0806, 'T': 0.1335, 'U': 0.1335, 'N': 0.0}
DPP_DIM = 16
STRUCTURE_TO_IDX = {'.': 0, '(': 1, ')': 2, 'N': 3}


def extract_5mer(seq, motif_size=5):
    seq = str(seq).strip().upper()
    length = len(seq)
    mid_idx = length // 2
    if length >= motif_size:
        half = motif_size // 2
        return seq[mid_idx - half: mid_idx + half + 1]
    return seq


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

    features = np.concatenate(all_features, axis=0)
    return features


def step1_group_by_motif():
    print("=" * 70)
    print("  Step 1: 读取CSV, 处理chr_context, 提取5-mer motif, 分组")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    print(f"  原始数据: {df.shape[0]} 行, {df.shape[1]} 列")

    df['chr_context'] = df['chr_context'].astype(str).str.upper().str.replace('U', 'T', regex=False)

    df['motif_5mer'] = df['chr_context'].apply(extract_5mer)

    with open(GROUP_INFO_PATH, 'r') as f:
        group_info = json.load(f)
    major_motifs = set(group_info['major_motifs'])
    print(f"  原有 major_motifs ({len(major_motifs)}): {sorted(major_motifs)}")

    def assign_group(motif):
        if motif in major_motifs:
            return motif
        return 'others'

    df['motif_group'] = df['motif_5mer'].apply(assign_group)

    group_counts = df['motif_group'].value_counts().sort_index()
    print(f"\n  各组样本数:")
    for g, c in group_counts.items():
        print(f"    {g}: {c}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(GROUPED_CSV_PATH, index=False)
    print(f"\n  分组后文件已保存: {GROUPED_CSV_PATH}")

    return df, major_motifs


def step2_stratified_sampling(df):
    print("\n" + "=" * 70)
    print("  Step 2: 分层抽样 ~10000 条, 各组尽量均衡")
    print("=" * 70)

    groups = df['motif_group'].unique()
    n_groups = len(groups)
    target_total = 10000
    per_group = target_total // n_groups

    print(f"  组数: {n_groups}, 目标总数: ~{target_total}, 每组目标: ~{per_group}")

    sampled_dfs = []
    rng = np.random.RandomState(SEED)

    for g in sorted(groups):
        g_df = df[df['motif_group'] == g]
        n_available = len(g_df)
        n_sample = min(per_group, n_available)
        sampled = g_df.sample(n=n_sample, random_state=rng)
        sampled_dfs.append(sampled)
        print(f"    {g}: 可用 {n_available}, 抽取 {n_sample}")

    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"\n  抽样后总条数: {len(sampled_df)}")
    group_counts = sampled_df['motif_group'].value_counts().sort_index()
    for g, c in group_counts.items():
        print(f"    {g}: {c}")

    sampled_df.to_csv(SAMPLED_CSV_PATH, index=False)
    print(f"\n  抽样文件已保存: {SAMPLED_CSV_PATH}")

    return sampled_df


def step3_predict(sampled_df):
    print("\n" + "=" * 70)
    print("  Step 3: 用训练好的模型预测各组序列")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    print("\n  [3.1] 加载融合模型...")
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
    print("  融合模型加载完成")

    tokenizer = AutoTokenizer.from_pretrained(DNABERT3_PATH)

    print("\n  [3.2] 加载基序分类器...")
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
    print(f"  加载 {len(classifiers)} 个分类器: {list(classifiers.keys())}")

    print("\n  [3.3] 提取1536维特征 (全部抽样数据)...")
    predict_df = sampled_df.copy()
    predict_df['text'] = predict_df['chr_context']

    dataset = FeatureExtractionDataset(predict_df, tokenizer, seq_len=201, max_len=256)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False,
                            collate_fn=collate_fn, num_workers=0,
                            pin_memory=device.type == 'cuda')

    all_features = extract_features(model, dataloader, device)
    print(f"  特征矩阵: {all_features.shape}")

    print("\n  [3.4] 逐组预测并保存结果...")
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    group_results = []

    groups = sorted(predict_df['motif_group'].unique())
    for g in groups:
        g_mask = predict_df['motif_group'].values == g
        g_indices = np.where(g_mask)[0]
        g_df = predict_df.iloc[g_indices].copy()
        g_features = all_features[g_indices]

        classifier_key = g if g in classifiers else 'others'
        if classifier_key not in classifiers:
            print(f"    {g}: 无对应分类器, 跳过")
            continue

        clf = classifiers[classifier_key]
        X = torch.tensor(g_features, dtype=torch.float32).to(device)

        all_probs = []
        for i in range(0, len(X), 256):
            batch = X[i:i + 256]
            with torch.no_grad():
                probs = clf(batch).cpu().numpy()
            all_probs.extend(probs)

        probs = np.array(all_probs)
        preds = (probs > 0.5).astype(int)

        g_df['pred_prob'] = probs
        g_df['pred_label'] = preds

        safe_name = g.replace('/', '_').replace('\\', '_')
        pred_path = os.path.join(PREDICTIONS_DIR, f"pred_{safe_name}.csv")
        g_df.to_csv(pred_path, index=False)

        n_total = len(g_df)
        n_pos = int(preds.sum())
        pos_ratio = n_pos / n_total if n_total > 0 else 0.0

        group_results.append({
            'motif_group': g,
            'classifier_used': classifier_key,
            'n_samples': n_total,
            'n_predicted_positive': n_pos,
            'positive_ratio': pos_ratio
        })
        print(f"    {g}: n={n_total}, 正样本预测={n_pos}, 比例={pos_ratio:.4f} -> {pred_path}")

    total_n = sum(r['n_samples'] for r in group_results)
    total_pos = sum(r['n_predicted_positive'] for r in group_results)
    overall_ratio = total_pos / total_n if total_n > 0 else 0.0

    overall_row = {
        'motif_group': 'OVERALL',
        'classifier_used': '-',
        'n_samples': total_n,
        'n_predicted_positive': total_pos,
        'positive_ratio': overall_ratio
    }

    print("\n" + "=" * 70)
    print("  正样本预测比例汇总")
    print("=" * 70)
    print(f"  {'组':<12} {'分类器':<12} {'样本数':>8} {'预测正样本':>10} {'正样本比例':>10}")
    print(f"  {'-'*56}")
    for r in group_results:
        print(f"  {r['motif_group']:<12} {r['classifier_used']:<12} {r['n_samples']:>8} "
              f"{r['n_predicted_positive']:>10} {r['positive_ratio']:>10.4f}")
    print(f"  {'-'*56}")
    print(f"  {'OVERALL':<12} {'-':<12} {total_n:>8} {total_pos:>10} {overall_ratio:>10.4f}")

    all_results = group_results + [overall_row]
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(POSITIVE_RATIO_PATH, index=False)
    print(f"\n  汇总文件已保存: {POSITIVE_RATIO_PATH}")
    print("=" * 70)

    return group_results, overall_ratio


def main():
    df, major_motifs = step1_group_by_motif()
    sampled_df = step2_stratified_sampling(df)
    group_results, overall_ratio = step3_predict(sampled_df)


if __name__ == "__main__":
    main()
