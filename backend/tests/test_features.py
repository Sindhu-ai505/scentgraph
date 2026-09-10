"""
Unit tests for Molecular Featurization and RDKit processing.
"""

import pytest
import torch
from backend.features import (
    smiles_to_graph,
    get_molecule_properties,
    render_molecule_png,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM
)


def test_smiles_to_graph_ethanol():
    # Ethanol: CCO (3 heavy atoms, 2 single bonds)
    data = smiles_to_graph("CCO")
    assert data is not None
    assert data.num_nodes == 3
    assert data.x.shape == (3, NODE_FEATURE_DIM)
    # 2 bonds -> 4 directed edges
    assert data.edge_index.shape == (2, 4)
    assert data.edge_attr.shape == (4, EDGE_FEATURE_DIM)


def test_smiles_to_graph_benzene():
    # Benzene: c1ccccc1 (6 heavy atoms, 6 aromatic bonds)
    data = smiles_to_graph("c1ccccc1")
    assert data is not None
    assert data.num_nodes == 6
    assert data.x.shape == (6, NODE_FEATURE_DIM)
    # 6 bonds -> 12 directed edges
    assert data.edge_index.shape == (2, 12)
    assert data.edge_attr.shape == (12, EDGE_FEATURE_DIM)


def test_smiles_to_graph_vanillin():
    # Vanillin: O=Cc1ccc(O)c(OC)c1 (11 heavy atoms)
    data = smiles_to_graph("O=Cc1ccc(O)c(OC)c1")
    assert data is not None
    assert data.num_nodes == 11
    assert data.x.shape == (11, NODE_FEATURE_DIM)
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM


def test_smiles_to_graph_invalid():
    assert smiles_to_graph("") is None
    assert smiles_to_graph("NOT_A_SMILES_STRING") is None
    assert smiles_to_graph("C123") is None


def test_get_molecule_properties():
    props = get_molecule_properties("CCO")
    assert props is not None
    assert props["formula"] == "C2H6O"
    assert props["heavy_atom_count"] == 3
    assert "molecular_weight" in props
    assert props["molecular_weight"] > 40.0


def test_render_molecule_png():
    img_uri = render_molecule_png("CCO")
    assert img_uri is not None
    assert img_uri.startswith("data:image/png;base64,")
    assert len(img_uri) > 100
