"""
visualize.py
------------
Loads the saved metrics.json from 'outputs/' and generates two plots:
    1. Training Loss curve
    2. Validation and Test PR-AUC curves

Run after train.py has completed at least one epoch.

Outputs:
    outputs/loss_curve.png    — Cross-entropy loss over epochs
    outputs/pr_auc_curve.png  — Validation and test PR-AUC over epochs
"""

import json
import os

import matplotlib.pyplot as plt

OUTPUT_DIR   = "outputs"
METRICS_FILE = f"{OUTPUT_DIR}/metrics.json"


def plot_loss(epochs, loss: list, save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label="Training Loss", color="darkred", lw=2)
    plt.title("GraphSAGE — Training Loss on Elliptic Dataset", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def plot_pr_auc(epochs, val_auc: list, test_auc: list, save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_auc,  label="Validation PR-AUC", color="teal",  lw=2)
    plt.plot(epochs, test_auc, label="Test PR-AUC",       color="navy",  lw=2, linestyle="--")
    plt.title("GraphSAGE — Precision-Recall AUC over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("PR-AUC Score", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def visualize():
    if not os.path.exists(METRICS_FILE):
        print(f"[ERROR] '{METRICS_FILE}' not found. Run train.py first.")
        return

    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)

    epochs = range(1, len(metrics["loss"]) + 1)

    plot_loss(epochs, metrics["loss"],        f"{OUTPUT_DIR}/loss_curve.png")
    plot_pr_auc(epochs, metrics["val_pr_auc"],
                metrics["test_pr_auc"],       f"{OUTPUT_DIR}/pr_auc_curve.png")

    print("[INFO] All plots saved to outputs/")


if __name__ == "__main__":
    visualize()