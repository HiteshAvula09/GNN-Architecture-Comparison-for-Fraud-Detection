# Graph Neural Network Architecture Comparison for Fraud Detection

This project provides a comprehensive comparative analysis of different Graph Neural Network (GNN) architectures for identifying fraudulent transactions within cryptocurrency networks. 

Using PyTorch Geometric, we model transactions as a massive graph and evaluate the performance of three popular and powerful GNN models: **GraphSAGE**, **GCN** (Graph Convolutional Networks), and **GAT** (Graph Attention Networks).

## Dataset

This project utilizes the [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
- **Nodes:** ~203,000 representing Bitcoin transactions.
- **Edges:** ~234,000 representing direct payment flows between transactions.
- **Features:** 166 features containing local structure and neighborhood information.
- **Goal:** Binary node classification to distinguish between "illicit" (fraudulent) and "licit" (normal) transactions.

The dataset is automatically downloaded and processed using standard Train, Validation, and Test splits. PyTorch Geometric handles the graph structures seamlessly.

## Implemented Architectures

We train and evaluate three baseline architectures to discover which inductive bias works best for transaction-based fraud:

1. **GraphSAGE (Baseline):** Samples and aggregates features from local neighborhoods. Highly scalable and generalized.
2. **GCN (Graph Convolutional Networks):** Uses an unweighted mean of all neighbors, approximating spectral graph convolutions.
3. **GAT (Graph Attention Networks):** Applies multi-head attention mechanisms to implicitly weigh the importance of different neighboring nodes during aggregation.

## Evaluation Metrics

Fraud datasets are heavily class-imbalanced. Because of this, standard Accuracy drops in reliability. Our pipeline tracks multiple robust classification metrics:
- **PR-AUC** (Precision-Recall Area Under the Curve) — The standard primary metric for this task.
- **Precision** & **Recall**
- **F1-Score**
- **Accuracy**
- **Confusion Matrices**

## Project Structure

- `requirements.txt`: Project dependencies (PyG, scikit-learn, etc.).
- `src/dataset.py`: Handles downloading the dataset and generating node masks.
- `src/compare_models.py`: The main orchestrator script. It defines all three GNN layers, trains them sequentially over 40 epochs, logs their losses, calculates the final evaluation metrics, and exports the data to JSON files and plots.
- `src/model.py`: Stores the baseline GraphSAGE architectural definition.
- `src/train.py` & `src/visualize.py`: Supporting scripts for training and visualizing a single baseline model independently.

## Getting Started

1. **Activate Environment**:
```bash
.venv\Scripts\activate
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Comparative Analysis**:
*(Note: The first execution securely downloads the raw dataset into the `data/` folder).*
```bash
python src/compare_models.py
```

## Results & Conclusion

After training the models on the dataset, the metrics clearly define **GraphSAGE** as the strongest performing setup under default hyperparameter constraints, achieving a stellar PR-AUC (~0.99) and F1-Score compared to GCN and GAT. 

GAT algorithms often require deeper hyperparameter tuning, more attention-heads, and extended epoch cycles to stabilize structure-based weights. For future homogeneous graph work on this dataset, GraphSAGE serves as the most reliable and expressive foundation.
