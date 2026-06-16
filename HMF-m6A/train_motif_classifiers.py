import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from motif_grouper import discover_motif_groups


class Config:
    FEATURES_PATH = "./features/m6a_features.npz"
    GROUPED_DATA_DIR = "./grouped_data"
    FEATURE_INDEX_MAP_PATH = "./grouped_data/feature_index_map.json"

    INPUT_DIM = 1536
    HIDDEN_DIMS = [512, 128]
    DROPOUT = 0.3

    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 7

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CLASSIFIERS_DIR = "./classifiers"
    ROUTING_MAP_PATH = "./classifiers/routing_map.json"


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


def load_group_data(group_dir: str):
    train_path = os.path.join(group_dir, 'train.tsv')
    val_path = os.path.join(group_dir, 'val.tsv')

    train_df = pd.read_csv(train_path, sep='\t') if os.path.exists(train_path) else pd.DataFrame()
    val_df = pd.read_csv(val_path, sep='\t') if os.path.exists(val_path) else pd.DataFrame()

    return train_df, val_df


def get_group_feature_indices(index_map: Dict, group_name: str) -> List[int]:
    return index_map.get(group_name, [])


def build_dataloader(features: np.ndarray, labels: np.ndarray,
                     indices: List[int], batch_size: int, shuffle: bool = True):
    if len(indices) == 0:
        return None

    X = torch.tensor(features[indices], dtype=torch.float32)
    y = torch.tensor(labels[indices], dtype=torch.float32)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader


def train_classifier(model, train_loader, val_loader, config, group_name: str):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    bce_fn = nn.BCELoss()

    best_auc = 0.0
    best_state = None
    best_metrics = {}
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(config.DEVICE)
            y_batch = y_batch.to(config.DEVICE)

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

        # ---- Validate ----
        if val_loader is None or len(val_loader.dataset) == 0:
            val_auc = train_auc
            val_loss = train_loss
            val_acc = accuracy_score(train_labels, train_preds)
            _, _, val_f1, _ = precision_recall_fscore_support(train_labels, train_preds, average='binary', zero_division=0)
        else:
            model.eval()
            val_loss = 0.0
            val_probs = []
            val_labels = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(config.DEVICE)
                    y_batch = y_batch.to(config.DEVICE)
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
            precision, recall, val_f1, _ = precision_recall_fscore_support(
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

        if patience_counter >= config.PATIENCE:
            break

    return best_auc, best_state, best_metrics


def main():
    config = Config()
    os.makedirs(config.CLASSIFIERS_DIR, exist_ok=True)

    print("=" * 70)
    print("  HMF Motif-specific classifier")
    print("  Architecture: 1536 -> 512 -> 128 -> 1 (sigmoid)")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")

    # ---- 1. Load features ----
    print("\n[1] Features file...")
    if not os.path.exists(config.FEATURES_PATH):
        print(f": Features file {config.FEATURES_PATH}")
        print("Please run extract_features_and_group.py first")
        return

    data = np.load(config.FEATURES_PATH, allow_pickle=True)
    features = data['features']
    labels = data['labels']
    motifs = data['motifs'] if 'motifs' in data else None
    print(f"  Feature matrix: {features.shape}, labels: {labels.shape}")

    # ---- 2. Load feature index map ----
    print("\n[2] Motif groups...")
    if not os.path.exists(config.FEATURE_INDEX_MAP_PATH):
        print(f"Error: index map not found {config.FEATURE_INDEX_MAP_PATH}")
        print("Please run extract_features_and_group.py first")
        return

    with open(config.FEATURE_INDEX_MAP_PATH, 'r') as f:
        index_map = json.load(f)

    group_names = discover_motif_groups(config.GROUPED_DATA_DIR)
    print(f"  Found {len(group_names)} Motif groups: {group_names}")

    # ---- 3. Motif ----
    print("\n[3] Motif-specific classifier...")
    routing_map = {}
    results = {}

    for group_name in group_names:
        print(f"\n{'-'*50}")
        print(f"  Motif: {group_name}")
        print(f"{'-'*50}")

        indices = index_map.get(group_name, [])
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

        # Train/val split (80/20, stratified)
        indices_arr = np.array(indices)
        pos_indices = indices_arr[group_labels == 1]
        neg_indices = indices_arr[group_labels == 0]

        np.random.seed(42)
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
            features, labels, train_indices, config.BATCH_SIZE, shuffle=True
        )
        val_loader = build_dataloader(
            features, labels, val_indices, config.BATCH_SIZE, shuffle=False
        )

        if train_loader is None:
            print(f"  Skip: empty training data")
            continue

        # Instantiate classifier
        model = MotifClassifier(
            input_dim=config.INPUT_DIM,
            hidden_dims=config.HIDDEN_DIMS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)

        # 
        best_auc, best_state, best_metrics = train_classifier(
            model, train_loader, val_loader, config, group_name
        )

        # Save classifier
        safe_name = group_name.replace('/', '_').replace('\\', '_')
        weight_path = os.path.join(config.CLASSIFIERS_DIR, f"classifier_{safe_name}.pt")
        torch.save({
            'model_state_dict': best_state,
            'input_dim': config.INPUT_DIM,
            'hidden_dims': config.HIDDEN_DIMS,
            'dropout': config.DROPOUT,
            'best_auc': best_auc,
            'best_acc': best_metrics.get('acc', 0.0),
            'best_f1': best_metrics.get('f1', 0.0),
            'group_name': group_name,
            'num_samples': len(indices),
            'pos_count': pos_count,
            'neg_count': neg_count
        }, weight_path)

        routing_map[group_name] = weight_path
        results[group_name] = {
            'auc': best_auc,
            'acc': best_metrics.get('acc', 0.0),
            'f1': best_metrics.get('f1', 0.0),
            'n_samples': len(indices),
            'pos': pos_count,
            'neg': neg_count
        }

        print(f"  Best AUC: {best_auc:.4f} Acc: {best_metrics.get('acc', 0.0):.4f} F1: {best_metrics.get('f1', 0.0):.4f}, saved: {weight_path}")

    # ---- 4. Routing map ----
    print("\n[4] Routing map...")
    with open(config.ROUTING_MAP_PATH, 'w') as f:
        json.dump({
            'routing': routing_map,
            'input_dim': config.INPUT_DIM,
            'hidden_dims': config.HIDDEN_DIMS,
            'dropout': config.DROPOUT,
            'default_group': 'others' if 'others' in routing_map else None
        }, f, indent=2)
    print(f"  Routing map: {config.ROUTING_MAP_PATH}")

    # ---- 5. Summary ----
    print("\n" + "=" * 70)
    print("  MotifTraining complete!")
    print(f"  Classifiers dir: {config.CLASSIFIERS_DIR}")
    print(f"  Routing map: {config.ROUTING_MAP_PATH}")
    print(f"\n  Per-group performance summary:")
    print(f"  {'Motif':<12} {'Samples':>8} {'pos/neg':>10} {'AUC':>8} {'ACC':>8} {'F1':>8}")
    print(f"  {'-'*58}")
    for name, info in sorted(results.items()):
        print(f"  {name:<12} {info['n_samples']:>8} {info['pos']:>4}/{info['neg']:<4} "
              f"{info['auc']:>8.4f} {info['acc']:>8.4f} {info['f1']:>8.4f}")

    if results:
        aucs = [v['auc'] for v in results.values()]
        accs = [v['acc'] for v in results.values()]
        f1s = [v['f1'] for v in results.values()]
        print(f"\n  Mean AUC: {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")
        print(f"  Mean ACC: {np.mean(accs):.4f} (±{np.std(accs):.4f})")
        print(f"  Mean F1:  {np.mean(f1s):.4f} (±{np.std(f1s):.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
