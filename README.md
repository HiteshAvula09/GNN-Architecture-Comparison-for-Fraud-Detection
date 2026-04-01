# 🔍 GNN Fraud Detection on Elliptic Bitcoin Dataset

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.x-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A comparative analysis of Graph Neural Network (GNN) architectures for detecting fraudulent Bitcoin transactions. This project benchmarks **GraphSAGE**, **GCN**, and **GAT** on the [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) — a real-world, large-scale cryptocurrency transaction graph.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architectures](#architectures)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Conclusion](#conclusion)

---

## Overview

Cryptocurrency fraud detection is a critical challenge in financial security. Traditional ML methods struggle because transaction data is inherently relational — money flows between wallets form a complex graph. GNNs can exploit this graph structure, leveraging the connections between transactions to improve classification.

This project:
- Models Bitcoin transactions as a homogeneous graph
- Trains three GNN architectures under identical conditions (40 epochs, same hyperparameters)
- Evaluates using **PR-AUC**, **F1-Score**, **Precision**, and **Recall** — metrics suited for imbalanced fraud datasets

---

## Dataset

| Property       | Value                             |
|----------------|-----------------------------------|
| **Source**     | Elliptic Bitcoin Dataset (Kaggle) |
| **Nodes**      | ~203,769 Bitcoin transactions     |
| **Edges**      | ~234,355 payment flows            |
| **Features**   | 166 per node (local + aggregated neighborhood) |
| **Classes**    | `0` = Illicit (fraud), `1` = Licit, `2` = Unknown (excluded) |
| **Split**      | Temporal train/test (built-in) + 20% dynamic validation |

> **Note:** The dataset is automatically downloaded via `torch_geometric.datasets.EllipticBitcoinDataset` on first run.

---

## Architectures

### 1. GraphSAGE *(Primary Baseline)*
Samples and aggregates features from local node neighborhoods. Highly scalable and effective on large graphs via inductive learning.

### 2. GCN — Graph Convolutional Network
Applies an unweighted mean aggregation across all neighbors, approximating spectral graph convolutions. Simple and efficient, but prone to over-smoothing on dense graphs.

### 3. GAT — Graph Attention Network
Uses multi-head attention to dynamically weight the importance of each neighboring node during aggregation. More expressive, but requires careful tuning to stabilize.

All three models share the same architecture template:
```
Input (166) → Conv Layer 1 (64, ReLU, Dropout 0.3) → Conv Layer 2 (64) → Output (2 classes)
```

---

## Results

All models trained for **40 epochs** | Optimizer: `Adam(lr=0.01, weight_decay=5e-4)` | Loss: `CrossEntropyLoss`

| Model         | PR-AUC | Accuracy | Precision | Recall | F1-Score |
|---------------|--------|----------|-----------|--------|----------|
| **GraphSAGE** | **0.9917** | **0.9487** | **0.9693** | **0.9759** | **0.9726** |
| GCN           | 0.9833 | 0.9099   | 0.9539    | 0.9496 | 0.9517   |
| GAT           | 0.9833 | 0.9051   | 0.9641    | 0.9332 | 0.9484   |

### Confusion Matrices

| Model       | TP    | FP  | FN  | TN     |
|-------------|-------|-----|-----|--------|
| GraphSAGE   | 602   | 481 | 375 | 15,212 |
| GCN         | 367   | 716 | 786 | 14,801 |
| GAT         | 542   | 541 | 1041| 14,546 |

> *TP = Illicit correctly detected. FN = Illicit missed (most costly error in fraud detection).*

---

## Project Structure

```
gnn-fraud-detection/
│
├── src/
│   ├── dataset.py          # Dataset loading, masking, and train/val/test splits
│   ├── model.py            # FraudGraphSAGE architecture definition
│   ├── train.py            # Single-model training loop with metric tracking
│   ├── visualize.py        # Loss and PR-AUC curve plotting
│   └── compare_models.py   # Full 3-model comparison pipeline
│
├── outputs/                # Generated at runtime
│   ├── best_model.pt       # Saved best GraphSAGE checkpoint
│   ├── metrics.json        # Training metrics history
│   ├── compare_metrics.json# Final metrics for all 3 models
│   ├── loss_curve.png      # Training loss plot
│   └── pr_auc_curve.png    # PR-AUC over epochs plot
│
├── data/                   # Auto-downloaded Elliptic dataset
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- CUDA-capable GPU *(recommended, CPU supported)*

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/gnn-fraud-detection.git
cd gnn-fraud-detection
```

### 2. Create & Activate a Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Full Model Comparison
```bash
python src/compare_models.py
```

> On first run, the Elliptic dataset (~50MB) will be automatically downloaded into `data/`.

### 5. (Optional) Train the Standalone GraphSAGE Baseline
```bash
python src/train.py
python src/visualize.py
```

---

## Conclusion

Under identical training conditions, **GraphSAGE outperforms GCN and GAT** across all key metrics on the Elliptic Bitcoin Dataset.

- **GCN** underperforms due to its unweighted mean aggregation, which over-smooths node representations across the dense transaction graph.
- **GAT** shows strong precision but trails in recall — its attention mechanism requires more epochs and careful hyperparameter tuning (heads, dropout, learning rate schedule) to fully converge.
- **GraphSAGE** benefits from neighborhood sampling and mean aggregation, making it both computationally efficient and highly effective for this task.

### Recommended Future Work
- **Temporal Splitting**: Respect the time-step ordering natively present in the Elliptic dataset to simulate real-world drift
- **GraphSMOTE**: Address class imbalance at the graph level via graph-based oversampling
- **Deeper GAT Tuning**: Increase attention heads, add layer normalization, and extend training epochs to unlock GAT's full expressivity
- **Heterogeneous Graph Modeling**: Introduce transaction-to-wallet edges for a richer relational structure

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.