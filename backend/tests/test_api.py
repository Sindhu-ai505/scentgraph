"""
Unit tests for FastAPI REST endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.api import app, load_assets


@pytest.fixture(scope="module", autouse=True)
def setup_api():
    load_assets()


client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "gatv2" in data["available_models"]
    assert data["vocab_size"] == 157


def test_get_labels():
    response = client.get("/labels")
    assert response.status_code == 200
    data = response.json()
    assert data["total_labels"] == 157
    assert len(data["labels"]) == 157
    # Verify first label is fruity or frequent
    assert data["labels"][0]["count"] >= 30
    assert "scent_family" in data["labels"][0]


def test_get_molecule_examples():
    response = client.get("/molecule/examples")
    assert response.status_code == 200
    data = response.json()
    assert len(data["molecules"]) >= 14
    first = data["molecules"][0]
    assert "name" in first
    assert "smiles" in first
    assert "image_png" in first
    assert first["image_png"].startswith("data:image/png;base64,")


def test_model_compare():
    response = client.get("/model/compare")
    assert response.status_code == 200
    data = response.json()
    assert "gatv2_summary" in data
    assert "gcn_summary" in data
    assert "comparison_table" in data
    assert len(data["comparison_table"]) == 157


def test_predict_valid_molecule():
    payload = {
        "smiles": "O=Cc1ccc(O)c(OC)c1",  # Vanillin
        "model": "gatv2"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["smiles"] == payload["smiles"]
    assert data["properties"]["formula"] == "C8H8O3"
    assert len(data["predictions"]) == 157
    assert data["image_png"].startswith("data:image/png;base64,")
    # Verify confidence is sorted descending
    confs = [p["confidence"] for p in data["predictions"]]
    assert confs == sorted(confs, reverse=True)


def test_predict_gcn():
    payload = {
        "smiles": "CC1=CCC(CC1)C(=C)C",  # Limonene
        "model": "gcn"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "gcn"
    assert len(data["predictions"]) == 157


def test_predict_invalid_molecule():
    payload = {
        "smiles": "NOT_A_SMILES",
        "model": "gatv2"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_predict_batch():
    payload = {
        "smiles_list": ["CCO", "c1ccccc1", "INVALID"],
        "model": "gatv2"
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert data["results"][0]["success"] is True
    assert data["results"][1]["success"] is True
    assert data["results"][2]["success"] is False
