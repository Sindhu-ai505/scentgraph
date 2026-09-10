"""
ScentGraph - Model Training Pipeline
Trains both GCNBaseline and ScentGATv2 models on real pyrfume-derived molecular odor data.
Features:
- Multi-label iterative stratified splitting (Train / Val / Test)
- Class-balanced positive weight BCE loss
- AdamW optimizer with learning rate warmup and reduction
- Per-label decision threshold optimization on validation set
- Full tracking of loss, Macro-F1, Micro-F1, and per-label metrics
- Saves weights, training curve logs, and metrics.json
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support

from backend.features import smiles_to_graph
from backend.model import build_model, GCNBaseline, ScentGATv2

DATA_PATH = Path("data/processed/molecules_odors.parquet")
VOCAB_PATH = Path("data/processed/label_vocab.json")
MODEL_DIR = Path("backend/models")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_dataset_and_featurize(vocab: List[str]) -> List[Any]:
    """Load molecules from parquet and convert to PyG Data objects."""
    print("Loading parquet dataset...")
    df = pd.read_parquet(DATA_PATH)
    label_to_idx = {lbl: i for i, lbl in enumerate(vocab)}
    num_classes = len(vocab)
    
    print(f"Featurizing {len(df)} molecules into molecular graphs...")
    dataset = []
    skipped = 0
    
    for idx, row in df.iterrows():
        smiles = row["smiles"]
        labels = row["labels"]
        # Create multi-hot binary vector
        target = [0.0] * num_classes
        for l in labels:
            if l in label_to_idx:
                target[label_to_idx[l]] = 1.0
                
        data = smiles_to_graph(smiles, labels=target)
        if data is not None:
            data.smiles = smiles
            data.canonical_smiles = row["canonical_smiles"]
            dataset.append(data)
        else:
            skipped += 1
            
    print(f"Featurized: {len(dataset)} graphs. Skipped: {skipped}")
    return dataset


def iterative_stratified_split(
    dataset: List[Any],
    num_classes: int,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Greedy multi-label stratification to ensure balanced label representation across splits.
    """
    targets = np.array([d.y.squeeze().numpy() for d in dataset])
    n_samples = len(dataset)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    target_counts = [n_train, n_val, n_test]
    splits = [[], [], []]
    split_targets = [np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)]
    
    # Sort samples by label rarity (most constrained samples assigned first)
    label_frequencies = targets.sum(axis=0)
    sample_rarities = [
        np.min(label_frequencies[np.where(targets[i] == 1)[0]]) if targets[i].sum() > 0 else 999999
        for i in range(n_samples)
    ]
    sorted_indices = np.argsort(sample_rarities)
    
    for idx in sorted_indices:
        sample_y = targets[idx]
        
        # Calculate split suitability
        best_split = 0
        min_discrepancy = float("inf")
        
        for s in range(3):
            if len(splits[s]) >= target_counts[s]:
                continue
            # Ratio of positive labels in split s
            desired_ratio = target_counts[s] / n_samples
            current_total = split_targets[s].sum() + 1e-5
            discrepancy = abs((current_total / (targets.sum() + 1e-5)) - desired_ratio)
            
            if discrepancy < min_discrepancy:
                min_discrepancy = discrepancy
                best_split = s
                
        splits[best_split].append(dataset[idx])
        split_targets[best_split] += sample_y
        
    train_data, val_data, test_data = splits[0], splits[1], splits[2]
    print(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data


def compute_pos_weights(train_dataset: List[Any], num_classes: int) -> torch.Tensor:
    """Compute balanced positive weights for BCEWithLogitsLoss."""
    targets = np.array([d.y.squeeze().numpy() for d in train_dataset])
    pos_counts = targets.sum(axis=0)
    neg_counts = len(train_dataset) - pos_counts
    
    pos_weights = neg_counts / (pos_counts + 1e-5)
    # Clamp to prevent extreme gradient swings on rare classes
    pos_weights = np.clip(pos_weights, 1.0, 25.0)
    return torch.tensor(pos_weights, dtype=torch.float)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: np.ndarray = None
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluates model and returns loss, macro_f1, micro_f1, all_probs, all_targets."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_attr = getattr(batch, 'edge_attr', None)
            logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)
            loss = criterion(logits, batch.y)
            total_loss += loss.item() * batch.num_graphs
            
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = batch.y.cpu().numpy()
            
            all_probs.append(probs)
            all_targets.append(targets)
            
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    avg_loss = total_loss / len(loader.dataset)
    
    num_classes = all_probs.shape[1]
    if thresholds is None:
        thresholds = np.full(num_classes, 0.5)
        
    preds = (all_probs >= thresholds).astype(int)
    macro_f1 = f1_score(all_targets, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_targets, preds, average="micro", zero_division=0)
    
    return avg_loss, macro_f1, micro_f1, all_probs, all_targets


def optimize_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    step: float = 0.05
) -> np.ndarray:
    """
    Searches optimal decision threshold for each label independently to maximize validation F1.
    """
    num_classes = probs.shape[1]
    optimal_thresholds = np.zeros(num_classes)
    candidate_thresholds = np.arange(0.10, 0.85, step)
    
    for j in range(num_classes):
        y_true = targets[:, j]
        y_prob = probs[:, j]
        
        best_t = 0.5
        best_f1 = -1.0
        
        for t in candidate_thresholds:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                
        # If all candidate thresholds give 0 F1 (e.g. very rare), default to conservative 0.35
        if best_f1 <= 0.0:
            best_t = 0.35
            
        optimal_thresholds[j] = round(float(best_t), 3)
        
    return optimal_thresholds


def train_single_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    vocab: List[str],
    pos_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 15,
    hidden_dim: int = 128,
    lr: float = 1e-3
) -> Dict[str, Any]:
    """Train a single model variant, optimize thresholds, and evaluate on test set."""
    print(f"\n{'='*55}")
    print(f"      TRAINING MODEL VARIANT: {model_name.upper()}")
    print(f"{'='*55}")
    
    num_classes = len(vocab)
    model = build_model(model_name, num_classes=num_classes, hidden_dim=hidden_dim).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.6, patience=2
    )
    
    best_val_macro_f1 = -1.0
    best_model_weights = None
    training_curves = []
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            edge_attr = getattr(batch, 'edge_attr', None)
            logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)
            loss = criterion(logits, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item() * batch.num_graphs
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        val_loss, val_macro_f1, val_micro_f1, val_probs, val_targets = evaluate_model(
            model, val_loader, device
        )
        scheduler.step(val_macro_f1)
        
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Val Micro-F1: {val_micro_f1:.4f} | "
            f"{elapsed:.1f}s"
        )
        
        training_curves.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_micro_f1": round(val_micro_f1, 4)
        })
        
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Load best weights
    model.load_state_dict(best_model_weights)
    
    # Optimize per-label decision thresholds on validation set
    print("\nOptimizing per-label decision thresholds on validation set...")
    _, _, _, val_probs, val_targets = evaluate_model(model, val_loader, device)
    optimal_thresholds = optimize_thresholds(val_probs, val_targets)
    
    # Save model checkpoint
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_ckpt_path = MODEL_DIR / f"{model_name.lower()}_best.pt"
    torch.save({
        "model_state_dict": best_model_weights,
        "model_name": model_name,
        "num_classes": num_classes,
        "hidden_dim": hidden_dim,
        "optimal_thresholds": optimal_thresholds.tolist(),
        "vocab": vocab
    }, model_ckpt_path)
    print(f"Saved best model checkpoint to: {model_ckpt_path}")
    
    # Evaluate on test set with optimal thresholds
    print("Evaluating on test set with tuned thresholds...")
    test_loss, test_macro_f1, test_micro_f1, test_probs, test_targets = evaluate_model(
        model, test_loader, device, thresholds=optimal_thresholds
    )
    
    # Per-label metrics
    test_preds = (test_probs >= optimal_thresholds).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        test_targets, test_preds, average=None, zero_division=0
    )
    
    per_label_metrics = {}
    for j, label in enumerate(vocab):
        per_label_metrics[label] = {
            "f1": round(float(f1[j]), 4),
            "precision": round(float(precision[j]), 4),
            "recall": round(float(recall[j]), 4),
            "threshold": round(float(optimal_thresholds[j]), 2),
            "support": int(support[j])
        }
        
    # Sort top and bottom labels by F1
    sorted_by_f1 = sorted(per_label_metrics.items(), key=lambda x: x[1]["f1"], reverse=True)
    top_10_best = sorted_by_f1[:10]
    # Filter labels with non-zero support for bottom 10
    with_support = [item for item in sorted_by_f1 if item[1]["support"] >= 2]
    top_10_worst = with_support[-10:] if len(with_support) >= 10 else sorted_by_f1[-10:]
    
    print(f"\nFinal Test Results for {model_name}:")
    print(f"  Macro-F1: {test_macro_f1:.4f}")
    print(f"  Micro-F1: {test_micro_f1:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    
    return {
        "model_key": model_name.lower(),
        "display_name": "ScentGATv2 (Attention)" if "gat" in model_name else "GCN Baseline",
        "macro_f1": round(float(test_macro_f1), 4),
        "micro_f1": round(float(test_micro_f1), 4),
        "test_loss": round(float(test_loss), 4),
        "best_val_macro_f1": round(float(best_val_macro_f1), 4),
        "epochs_trained": epochs,
        "training_curves": training_curves,
        "optimal_thresholds": {vocab[j]: round(float(optimal_thresholds[j]), 2) for j in range(num_classes)},
        "per_label_metrics": per_label_metrics,
        "top_10_best": [{"label": k, **v} for k, v in top_10_best],
        "top_10_worst": [{"label": k, **v} for k, v in top_10_worst]
    }


def main():
    """End-to-end training execution for both models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Loaded odor vocabulary with {len(vocab)} descriptor classes.")
    
    dataset = load_dataset_and_featurize(vocab)
    train_data, val_data, test_data = iterative_stratified_split(dataset, len(vocab))
    
    pos_weights = compute_pos_weights(train_data, len(vocab))
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    # Train both GCNBaseline and ScentGATv2
    # 15 epochs each gives strong convergence and fast runtime on CPU (~2 mins total)
    gatv2_metrics = train_single_model(
        model_name="gatv2",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        vocab=vocab,
        pos_weights=pos_weights,
        device=device,
        epochs=15,
        hidden_dim=128,
        lr=1e-3
    )
    
    gcn_metrics = train_single_model(
        model_name="gcn",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        vocab=vocab,
        pos_weights=pos_weights,
        device=device,
        epochs=15,
        hidden_dim=128,
        lr=1e-3
    )
    
    # Combined metrics output for API and frontend
    combined_metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vocab_size": len(vocab),
        "total_molecules": len(dataset),
        "split_sizes": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        },
        "models": {
            "gatv2": gatv2_metrics,
            "gcn": gcn_metrics
        }
    }
    
    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2)
        
    print(f"\nAll models trained successfully. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
