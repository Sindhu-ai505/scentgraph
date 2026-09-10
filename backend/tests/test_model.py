"""
Unit tests for GNN Architectures (GCNBaseline and ScentGATv2).
"""

import pytest
import torch
from torch_geometric.data import Data, Batch
from backend.features import smiles_to_graph, NODE_FEATURE_DIM, EDGE_FEATURE_DIM
from backend.model import build_model, GCNBaseline, ScentGATv2


def test_build_model():
    gcn = build_model("gcn", num_classes=157, hidden_dim=64)
    assert isinstance(gcn, GCNBaseline)

    gat = build_model("gatv2", num_classes=157, hidden_dim=64)
    assert isinstance(gat, ScentGATv2)


def test_gcn_forward():
    model = GCNBaseline(node_in_dim=NODE_FEATURE_DIM, hidden_dim=64, num_classes=157)
    g1 = smiles_to_graph("CCO")
    g2 = smiles_to_graph("c1ccccc1")
    batch = Batch.from_data_list([g1, g2])

    logits = model(batch.x, batch.edge_index, batch.batch)
    assert logits.shape == (2, 157)


def test_gatv2_forward():
    model = ScentGATv2(
        node_in_dim=NODE_FEATURE_DIM,
        edge_in_dim=EDGE_FEATURE_DIM,
        hidden_dim=64,
        num_classes=157,
        num_layers=2
    )
    g1 = smiles_to_graph("CCO")
    g2 = smiles_to_graph("c1ccccc1")
    batch = Batch.from_data_list([g1, g2])

    logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
    assert logits.shape == (2, 157)
