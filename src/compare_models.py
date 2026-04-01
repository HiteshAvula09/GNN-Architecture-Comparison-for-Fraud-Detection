"""
compare_models.py
-----------------
Trains and evaluates three GNN architectures side-by-side on the Elliptic
Bitcoin Dataset:
    - GraphSAGE
    - GCN  (Graph Convolutional Network)
    - GAT  (Graph Attention Network)

All models share the same hyperparameters and training duration (40 epochs)
for a fair comparison. Final metrics are printed and saved to JSON.

Outputs:
    outputs/compare_metrics.json — Final evaluation metrics for all three models
"""

import json
import os

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from dataset import get_elliptic_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EPOCHS          = 40
LEARNING_RATE   = 0.01
WEIGHT_DECAY    = 5e-4
HIDDEN_CHANNELS = 64
OUTPUT_DIR      = "outputs"


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

class GraphSAGEModel(torch.nn.Module):
    """Two-layer GraphSAGE with mean aggregation."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)


class GCNModel(torch.nn.Module):
    """Two-layer GCN with symmetric normalization."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)


class GATModel(torch.nn.Module):
    """Two-layer GAT with 4 attention heads on the first layer."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_and_eval(model_name: str, model, data, train_mask, test_mask, device):
    """
    Runs the full training loop and returns PR-AUC history + final metrics.

    Args:
        model_name  (str):  Display name of the model.
        model       (nn.Module): The GNN model to train.
        data        (Data): PyG graph data object.
        train_mask  (Tensor): Boolean mask for training nodes.
        test_mask   (Tensor): Boolean mask for test nodes.
        device      (torch.device): Training device.

    Returns:
        pr_auc_history (list[float]): Test PR-AUC at each epoch.
        final_metrics  (dict): Evaluation metrics at final epoch.
    """
    print(f"\n[INFO] Training {model_name}...")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss()

    pr_auc_history = []
    final_metrics  = {}

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # --- Evaluate ---
        model.eval()
        with torch.no_grad():
            out   = model(data.x, data.edge_index)
            probs = torch.softmax(out, dim=1)[:, 0]  # P(illicit)
            preds = out.argmax(dim=1)

            y_true = (data.y[test_mask] == 0).cpu().numpy()
            y_prob = probs[test_mask].cpu().numpy()
            y_pred = (preds[test_mask] == 0).cpu().numpy()

            test_auc = average_precision_score(y_true, y_prob)
            pr_auc_history.append(test_auc)

        # --- Final epoch metrics ---
        if epoch == EPOCHS:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            acc = accuracy_score(y_true, y_pred)
            cm  = confusion_matrix(y_true, y_pred)

            final_metrics = {
                "PR_AUC":           round(test_auc,  4),
                "Accuracy":         round(acc,       4),
                "Precision":        round(precision, 4),
                "Recall":           round(recall,    4),
                "F1":               round(f1,        4),
                "Confusion_Matrix": cm.tolist(),
            }

            print(f"\n  [{model_name}] Final Results @ Epoch {EPOCHS}")
            print(f"  {'─' * 38}")
            print(f"  Loss:             {loss:.4f}")
            for k, v in final_metrics.items():
                print(f"  {k:<18}: {v}")

    return pr_auc_history, final_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data, train_mask, val_mask, test_mask, num_classes = get_elliptic_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    data = data.to(device)

    in_channels = data.x.shape[1]

    models = {
        "GraphSAGE": GraphSAGEModel(in_channels, HIDDEN_CHANNELS, num_classes).to(device),
        "GCN":       GCNModel      (in_channels, HIDDEN_CHANNELS, num_classes).to(device),
        "GAT":       GATModel      (in_channels, HIDDEN_CHANNELS, num_classes).to(device),
    }

    all_metrics = {}

    for name, model in models.items():
        _, all_metrics[name] = train_and_eval(
            name, model, data, train_mask, test_mask, device
        )

    # --- Save results ---
    out_path = f"{OUTPUT_DIR}/compare_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n[INFO] Metrics saved to '{out_path}'")

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("  FINAL MODEL COMPARISON")
    print("=" * 60)
    header = f"  {'Model':<12} {'PR-AUC':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}"
    print(header)
    print("  " + "-" * 56)
    for name, m in all_metrics.items():
        print(
            f"  {name:<12} "
            f"{m['PR_AUC']:>8.4f} "
            f"{m['Accuracy']:>8.4f} "
            f"{m['Precision']:>8.4f} "
            f"{m['Recall']:>8.4f} "
            f"{m['F1']:>8.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()