"""
train.py
--------
Trains the FraudGraphSAGE model on the Elliptic Bitcoin Dataset and saves
the best checkpoint and metric history to the 'outputs/' directory.

Outputs:
    outputs/best_model.pt   — Model weights at peak validation PR-AUC
    outputs/metrics.json    — Per-epoch loss, val PR-AUC, and test PR-AUC
"""

import json
import os

import torch
from sklearn.metrics import average_precision_score

from dataset import get_elliptic_dataset
from model import FraudGraphSAGE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EPOCHS         = 40
LEARNING_RATE  = 0.01
WEIGHT_DECAY   = 5e-4
HIDDEN_CHANNELS = 64
OUTPUT_DIR     = "outputs"


def compute_pr_auc(model_out: torch.Tensor, data, mask) -> float:
    """Compute Precision-Recall AUC for the illicit class (class 0)."""
    probs  = torch.softmax(model_out, dim=1)[:, 0]
    y_true = (data.y[mask] == 0).cpu().numpy()
    y_prob = probs[mask].cpu().numpy()
    return average_precision_score(y_true, y_prob)


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Data ---
    data, train_mask, val_mask, test_mask, num_classes = get_elliptic_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    data = data.to(device)

    # --- Model ---
    model = FraudGraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss()

    # --- Training Loop ---
    best_val_auc = 0.0
    history = {"loss": [], "val_pr_auc": [], "test_pr_auc": []}

    print(f"[INFO] Training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        # Forward + backward pass
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            out      = model(data.x, data.edge_index)
            val_auc  = compute_pr_auc(out, data, val_mask)
            test_auc = compute_pr_auc(out, data, test_mask)

        history["loss"].append(loss.item())
        history["val_pr_auc"].append(val_auc)
        history["test_pr_auc"].append(test_auc)

        # Save best checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:03d} | "
                f"Loss: {loss:.4f} | "
                f"Val PR-AUC: {val_auc:.4f} | "
                f"Test PR-AUC: {test_auc:.4f}"
            )

    # --- Save Metrics ---
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(history, f, indent=4)

    print(f"\n[INFO] Training complete.")
    print(f"[INFO] Best Val PR-AUC: {best_val_auc:.4f}")
    print(f"[INFO] Outputs saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    train()