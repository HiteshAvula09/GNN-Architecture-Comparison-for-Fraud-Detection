"""
dataset.py
----------
Handles downloading, processing, and splitting the Elliptic Bitcoin Dataset
using PyTorch Geometric's built-in dataset wrapper.

Dataset Summary:
    - Nodes:    ~203,769 Bitcoin transactions
    - Edges:    ~234,355 directed payment flows
    - Features: 166 (local + aggregated neighborhood)
    - Classes:  0 = Illicit (fraud), 1 = Licit, 2 = Unknown (excluded)
"""

import torch
from torch_geometric.datasets import EllipticBitcoinDataset


def get_elliptic_dataset(root: str = "data"):
    """
    Downloads (on first run) and returns the Elliptic Bitcoin Dataset
    with train, validation, and test masks.

    The dataset uses a built-in temporal train/test split. Unlabelled nodes
    (class 2) are excluded from all masks. A 20% validation set is carved
    out of the training split using a fixed random seed for reproducibility.

    Args:
        root (str): Directory to store the downloaded dataset. Default: 'data'.

    Returns:
        data        (torch_geometric.data.Data): The full graph object.
        train_mask  (torch.BoolTensor): Mask for training nodes.
        val_mask    (torch.BoolTensor): Mask for validation nodes (20% of train).
        test_mask   (torch.BoolTensor): Mask for test nodes.
        num_classes (int): Number of target classes (2: illicit vs licit).
    """
    print(f"[INFO] Loading Elliptic Bitcoin Dataset from '{root}'...")
    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]

    num_classes = 2  # Binary: illicit (0) vs licit (1)

    print(f"[INFO] Nodes: {data.num_nodes:,} | Edges: {data.num_edges:,}")
    print(f"[INFO] Node features: {dataset.num_node_features}")

    # --- Masking ---
    # Exclude unlabelled nodes (class 2) from all splits
    labeled_mask = (data.y == 0) | (data.y == 1)

    train_mask = data.train_mask.bool() & labeled_mask
    test_mask  = data.test_mask.bool()  & labeled_mask

    # Carve out a 20% validation set from training indices
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    num_val = int(len(train_indices) * 0.2)

    torch.manual_seed(42)
    perm = torch.randperm(len(train_indices))
    val_indices          = train_indices[perm[:num_val]]
    train_indices_final  = train_indices[perm[num_val:]]

    val_mask = torch.zeros_like(train_mask)
    val_mask[val_indices] = True

    train_mask_final = torch.zeros_like(train_mask)
    train_mask_final[train_indices_final] = True

    print(f"[INFO] Train nodes: {train_mask_final.sum().item():,}")
    print(f"[INFO] Val nodes:   {val_mask.sum().item():,}")
    print(f"[INFO] Test nodes:  {test_mask.sum().item():,}")

    return data, train_mask_final, val_mask, test_mask, num_classes


if __name__ == "__main__":
    get_elliptic_dataset()