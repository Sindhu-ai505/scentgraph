"""
ScentGraph - Graph Neural Network Architectures
Implements two distinct GNN models:
1. GCNBaseline: Standard GCN convolutions with Mean/Max Pooling
2. ScentGATv2: Multi-round GATv2Conv message passing with edge conditioning
              and learned Global Attention Pooling.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATv2Conv,
    global_mean_pool,
    global_max_pool,
    GlobalAttention
)

from backend.features import NODE_FEATURE_DIM, EDGE_FEATURE_DIM


class GCNBaseline(nn.Module):
    """
    Graph Convolutional Network Baseline.
    Applies 3 rounds of GCN convolutions followed by concatenated Mean & Max pooling
    and an MLP multi-label classification head.
    """
    def __init__(
        self,
        node_in_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = 128,
        num_classes: int = 157,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        self.dropout = nn.Dropout(dropout)
        
        # Concatenated mean and max pooling -> 2 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.node_embed(x)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            residual = h
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
            h = h + residual  # Skip connection
            
        # Global pooling
        mean_p = global_mean_pool(h, batch)
        max_p = global_max_pool(h, batch)
        pooled = torch.cat([mean_p, max_p], dim=1)
        
        # Raw logits for BCEWithLogitsLoss
        logits = self.classifier(pooled)
        return logits


class ScentGATv2(nn.Module):
    """
    Advanced Message-Passing GNN using GATv2Conv layers with bond edge attributes
    and Global Attention Pooling to weight key odor-inducing functional groups.
    """
    def __init__(
        self,
        node_in_dim: int = NODE_FEATURE_DIM,
        edge_in_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 128,
        num_classes: int = 157,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim // 2)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False,
                edge_dim=hidden_dim // 2,
                dropout=dropout
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        self.dropout = nn.Dropout(dropout)
        
        # Learned Global Attention Pooling: gate_nn computes atom importance
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.pool = GlobalAttention(gate_nn=gate_nn)
        
        # Multi-layer Perceptron classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        h = self.node_embed(x)
        
        if edge_attr is not None and edge_attr.size(0) > 0:
            e = self.edge_embed(edge_attr)
        else:
            e = None
            
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            h = conv(h, edge_index, edge_attr=e)
            h = norm(h)
            h = F.gelu(h)
            h = self.dropout(h)
            h = h + residual  # Residual connection
            
        # Attention-based global molecular embedding
        mol_embedding = self.pool(h, batch)
        
        # Classification logits
        logits = self.classifier(mol_embedding)
        return logits


def build_model(
    model_name: str,
    num_classes: int = 157,
    hidden_dim: int = 128,
    dropout: float = 0.2
) -> nn.Module:
    """Model factory helper."""
    name = model_name.lower().strip()
    if "gat" in name or "nnconv" in name:
        return ScentGATv2(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    elif "gcn" in name:
        return GCNBaseline(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_name}. Choose 'gatv2' or 'gcn'.")
