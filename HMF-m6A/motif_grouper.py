import os
import json
import pandas as pd
from collections import Counter, defaultdict


def extract_5mer(seq, motif_size=5):
    seq = str(seq).strip().upper()
    length = len(seq)
    mid_idx = length // 2
    if length >= motif_size:
        half = motif_size // 2
        return seq[mid_idx - half : mid_idx + half + 1]
    return seq


def analyze_motifs_by_positive(df_full, threshold_ratio=0.015):
    print("\n[Motif] ====== Positive-Sample-Based Motif Grouping ======")

    df = df_full.copy()
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else df.columns[1]
    df = df.rename(columns={text_col: 'text', label_col: 'label'})

    df['motif'] = df['text'].apply(lambda x: extract_5mer(x, motif_size=5))

    invalid_mask = df['motif'].isnull() | (df['motif'].str.len() != 5)
    if invalid_mask.sum() > 0:
        print(f"[Motif] Warning: {invalid_mask.sum()} samples have invalid motifs, dropping")
        df = df[~invalid_mask]

    total_all = len(df)
    pos_df = df[df['label'] == 1].copy()
    neg_df = df[df['label'] == 0].copy()
    total_pos = len(pos_df)
    total_neg = len(neg_df)

    print(f"[Motif] Total samples: {total_all} (pos:{total_pos} / neg:{total_neg})")

    threshold_count = int(total_pos * threshold_ratio)
    print(f"[Motif] Positive count: {total_pos}, threshold ratio: {threshold_ratio}")
    print(f"[Motif] Grouping threshold (positive * {threshold_ratio}): {threshold_count}")

    pos_motif_counts = Counter(pos_df['motif'])
    all_motif_counts = Counter(df['motif'])

    major_motifs_set = set()
    minor_motifs_in_pos = []
    other_pos_count = 0

    for motif, pos_cnt in pos_motif_counts.most_common():
        if pos_cnt >= threshold_count:
            major_motifs_set.add(motif)
        else:
            minor_motifs_in_pos.append((motif, pos_cnt))
            other_pos_count += pos_cnt

    major_motifs = sorted(major_motifs_set)

    print(f"\n[Motif] Major motifs (pos count >= {threshold_count}): {len(major_motifs)}")
    print(f"[Motif] Minor motifs (-> others): {len(minor_motifs_in_pos)}, "
          f"involving {other_pos_count} positive samples")

    print(f"\n[Motif] === Major motifs detail ===")
    for m in sorted(major_motifs):
        pc = pos_motif_counts[m]
        ac = all_motif_counts[m]
        print(f"  {m}: pos={pc}, total={ac} ({ac/total_all*100:.2f}%)")

    if minor_motifs_in_pos:
        print(f"\n[Motif] === Minor motifs (top 10 -> others) ===")
        for m, c in sorted(minor_motifs_in_pos, key=lambda x: -x[1])[:10]:
            ac = all_motif_counts[m]
            print(f"  {m}: pos={c}, total={ac}")

    return df, major_motifs, other_pos_count > 0


def group_data_by_motif(df_with_motif, major_motifs, has_others, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    motif_groups = defaultdict(list)

    for idx, row in df_with_motif.iterrows():
        motif = row['motif']
        if motif in major_motifs:
            group_key = motif
        else:
            group_key = 'others'
        motif_groups[group_key].append({
            'text': row['text'],
            'label': int(row['label']),
            'motif': motif
        })

    group_names = []
    for motif_name in sorted(motif_groups.keys()):
        group_dir = os.path.join(output_dir, motif_name)
        os.makedirs(group_dir, exist_ok=True)

        group_df = pd.DataFrame(motif_groups[motif_name])

        if len(group_df) >= 4:
            train_df = group_df.sample(frac=0.8, random_state=42)
            val_df = group_df.drop(train_df.index)
        else:
            train_df = group_df
            val_df = pd.DataFrame(columns=group_df.columns)

        train_path = os.path.join(group_dir, 'train.tsv')
        val_path = os.path.join(group_dir, 'val.tsv')

        train_df.to_csv(train_path, sep='\t', index=False)
        val_df.to_csv(val_path, sep='\t', index=False)

        pos_cnt = int((group_df['label'] == 1).sum())
        neg_cnt = int((group_df['label'] == 0).sum())
        print(f"  [{motif_name}] n={len(group_df)} (+{pos_cnt}/-{neg_cnt}) | Train:{len(train_df)} Val:{len(val_df)}")
        group_names.append(motif_name)

    info_path = os.path.join(output_dir, 'group_info.json')
    with open(info_path, 'w') as f:
        json.dump({
            'groups': sorted(group_names),
            'num_groups': len(group_names),
            'major_motifs': sorted(major_motifs),
            'has_others': has_others,
            'grouping_strategy': 'positive_sample_based',
            'threshold_ratio': 0.015
        }, f, indent=2)

    print(f"\n[Motif] Grouped data saved to: {output_dir}")
    print(f"[Motif] Total groups created: {len(group_names)} (including {'others' if has_others else 'no'} others)")
    return group_names


def run_grouping(config):
    data_path = config.DATA_PATH
    grouped_data_dir = config.GROUPED_DATA_DIR

    print("=" * 70)
    print("  Motif Grouping Pipeline (Positive-Sample-Based)")
    print(f"  Data: {data_path}")
    print(f"  Strategy: threshold = num_positive_samples * 0.015")
    print(f"  Output dir: {grouped_data_dir}")
    print("=" * 70)

    df_full = pd.read_csv(data_path, sep='\t')

    df_with_motif, major_motifs, has_others = analyze_motifs_by_positive(
        df_full, threshold_ratio=config.MOTIF_THRESHOLD_RATIO
    )

    group_names = group_data_by_motif(df_with_motif, major_motifs, has_others, grouped_data_dir)
    return group_names


def discover_motif_groups(grouped_data_dir):
    motif_dirs = []
    if not os.path.exists(grouped_data_dir):
        return motif_dirs
    for name in sorted(os.listdir(grouped_data_dir)):
        dir_path = os.path.join(grouped_data_dir, name)
        if os.path.isdir(dir_path):
            train_file = os.path.join(dir_path, 'train.tsv')
            if os.path.exists(train_file):
                motif_dirs.append(name)
    return motif_dirs
