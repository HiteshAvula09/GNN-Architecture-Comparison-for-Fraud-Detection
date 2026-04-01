"""
model.py
--------
Defines the FraudGraphSAGE model — a two-layer GraphSAGE network for
binary node classification on the Elliptic Bitcoin Dataset.

GraphSAGE learns by sampling and aggregating features from local neighborhoods,
making it highly scalable and effective for large, inductive graph tasks.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FraudGraphSAGE(torch.nn.Module):
    """
    Two-layer GraphSAGE network for fraud transaction classification.

    Architecture:
        Input (166) → SAGEConv(64) → ReLU → Dropout
                    → SAGEConv(64) → ReLU → Dropout
                    → Linear(out_channels)

    Args:
        in_channels     (int): Number of input node features.
        hidden_channels (int): Size of hidden representations.
        out_channels    (int): Number of output classes (2 for binary classification).
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(FraudGraphSAGE, self).__init__()

        self.conv1   = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2   = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.out     = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1: aggregate + activate
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2: aggregate + activate
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Output logits
        return self.out(x)