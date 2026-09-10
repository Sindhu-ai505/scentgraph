"""
ScentGraph - FastAPI REST Backend
Provides endpoints for molecule scent prediction, batch scoring,
model metric comparison, label exploration, and curated aroma molecules.
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

from backend.features import (
    smiles_to_graph,
    get_molecule_properties,
    render_molecule_png
)
from backend.model import build_model, GCNBaseline, ScentGATv2

# Paths
VOCAB_PATH = Path("data/processed/label_vocab.json")
STATS_PATH = Path("data/processed/dataset_stats.json")
METRICS_PATH = Path("backend/models/metrics.json")
MODEL_DIR = Path("backend/models")

app = FastAPI(
    title="ScentGraph API",
    description="Molecule-to-Scent Predictor powered by Graph Neural Networks",
    version="1.0.0"
)

# CORS Middleware for local and production frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cached resources
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB: List[str] = []
DATASET_STATS: Dict[str, Any] = {}
MODELS_CACHE: Dict[str, Any] = {}
THRESHOLDS_CACHE: Dict[str, Dict[str, float]] = {}

# Scent family grouping
SCENT_FAMILIES = {
    "floral": ["floral", "rose", "jasmine", "violet", "lily", "lilac", "gardenia", "iris", "hyacinth", "orange blossom", "blossom", "carnation", "hawthorn", "lavender", "magnolia", "narcissus", "ylang"],
    "woody": ["woody", "cedar", "sandalwood", "pine", "vetiver", "patchouli", "bark", "moss", "oakmoss", "coniferous", "fir", "resinous", "amber", "terpenic"],
    "citrus": ["citrus", "lemon", "orange", "lime", "grapefruit", "bergamot", "mandarin", "peel", "zest"],
    "fruity": ["fruity", "apple", "banana", "berry", "strawberry", "cherry", "peach", "pear", "pineapple", "grape", "melon", "apricot", "plum", "mango", "tropical", "coconut", "fig", "guava", "passion fruit", "kiwi", "papaya"],
    "sweet": ["sweet", "caramel", "vanilla", "honey", "chocolate", "cocoa", "creamy", "buttery", "sugary", "jammy", "malt", "molasses", "brown sugar", "coumarin"],
    "green": ["green", "herbal", "grass", "leaf", "mint", "minty", "rosemary", "thyme", "eucalyptus", "tea", "vegetable", "celery", "tomato", "pea", "clary sage", "foliage", "fresh cut grass"],
    "spicy": ["spicy", "clove", "cinnamon", "pepper", "warm", "balsamic", "anise", "nutmeg", "ginger", "cardamom", "allspice", "peppery", "curry"],
    "sulfurous": ["sulfur", "meaty", "roasted", "onion", "garlic", "savory", "broth", "bacon", "coffee", "burnt", "smoky", "toast", "gourmand", "nutty", "almond", "peanut"],
    "earthy": ["earthy", "musk", "animal", "leather", "tobacco", "phenolic", "medicinal", "dirty", "damp", "musty", "hay", "mossy"],
    "fresh": ["fresh", "ethereal", "cooling", "clean", "solvent", "alcoholic", "ozone", "watery", "aquatic", "soap", "waxy", "fatty", "oily", "aldehydic", "camphor", "sharp"]
}


def get_scent_family(label: str) -> str:
    """Classifies an odor label into its primary fragrance family."""
    lbl = label.lower().strip()
    for family, members in SCENT_FAMILIES.items():
        if lbl in members:
            return family
        for m in members:
            if m in lbl or lbl in m:
                return family
    return "other"


CURATED_MOLECULES = [
    {
        "name": "Vanillin",
        "smiles": "O=Cc1ccc(O)c(OC)c1",
        "formula": "C8H8O3",
        "scent_notes": ["vanilla", "sweet", "creamy", "caramel"],
        "category": "sweet",
        "description": "The primary component of the extract of vanilla bean, widely used in perfumery and food flavoring."
    },
    {
        "name": "Limonene",
        "smiles": "CC1=CCC(CC1)C(=C)C",
        "formula": "C10H16",
        "scent_notes": ["citrus", "orange", "fresh", "lemon"],
        "category": "citrus",
        "description": "Abundant in citrus fruit rinds, giving oranges and lemons their signature bright aroma."
    },
    {
        "name": "Geraniol",
        "smiles": "CC(C)=CCCC(C)=CCO",
        "formula": "C10H18O",
        "scent_notes": ["floral", "rose", "sweet", "fruity"],
        "category": "floral",
        "description": "A primary component of rose oil and palmarosa oil with an elegant floral bouquet."
    },
    {
        "name": "Cinnamaldehyde",
        "smiles": "O=C/C=C/c1ccccc1",
        "formula": "C9H8O",
        "scent_notes": ["spicy", "cinnamon", "warm", "sweet"],
        "category": "spicy",
        "description": "The organic compound that imparts the characteristic pungent, warm spice flavor of cinnamon."
    },
    {
        "name": "Benzaldehyde",
        "smiles": "O=Cc1ccccc1",
        "formula": "C7H6O",
        "scent_notes": ["almond", "cherry", "sweet", "nutty"],
        "category": "fruity",
        "description": "Famous for its marzipan, almond, and cherry aroma, occurring naturally in bitter almonds."
    },
    {
        "name": "Menthol",
        "smiles": "CC1CCC(C(C1)O)C(C)C",
        "formula": "C10H20O",
        "scent_notes": ["minty", "cooling", "fresh", "herbal"],
        "category": "fresh",
        "description": "Derived from peppermint oils, delivering an intense cooling, minty sensation."
    },
    {
        "name": "Eugenol",
        "smiles": "COc1cc(CC=C)ccc1O",
        "formula": "C10H12O2",
        "scent_notes": ["spicy", "clove", "woody", "warm"],
        "category": "spicy",
        "description": "Extracted from clove oil, characterized by an aromatic, spicy-medicinal clove bouquet."
    },
    {
        "name": "Linalool",
        "smiles": "CC(C)=CCCC(C)(O)C=C",
        "formula": "C10H18O",
        "scent_notes": ["floral", "citrus", "lavender", "sweet"],
        "category": "floral",
        "description": "Found in over 200 plant species including lavender and basil, contributing a gentle floral-woody nuance."
    },
    {
        "name": "Isoamyl Acetate",
        "smiles": "CC(=O)OCCC(C)C",
        "formula": "C7H14O2",
        "scent_notes": ["fruity", "banana", "sweet", "apple"],
        "category": "fruity",
        "description": "Known as 'banana oil', widely used in confections for its authentic sweet banana essence."
    },
    {
        "name": "beta-Damascenone",
        "smiles": "CC1=C(C(C=CC1(C)C)C(=O)/C=C/C)C",
        "formula": "C13H18O",
        "scent_notes": ["rose", "apple", "fruity", "floral"],
        "category": "floral",
        "description": "An exceptionally potent rose ketone with an odor threshold under 2 picograms/liter."
    },
    {
        "name": "Geosmin",
        "smiles": "CC1CCC2(C)CCCC(O)C12",
        "formula": "C12H22O",
        "scent_notes": ["earthy", "musty", "soil", "fresh"],
        "category": "earthy",
        "description": "The biological compound responsible for the scent of petrichor (rain hitting dry soil)."
    },
    {
        "name": "Coumarin",
        "smiles": "O=C1OC2=CC=CC=C2C=C1",
        "formula": "C9H6O2",
        "scent_notes": ["sweet", "tonka", "vanilla", "hay"],
        "category": "sweet",
        "description": "A sweet, vanilla-like aroma reminiscent of fresh cut sweetgrass and tonka beans."
    },
    {
        "name": "2-Isobutyl-3-methoxypyrazine",
        "smiles": "COc1ncc(CC(C)C)nc1",
        "formula": "C9H14N2O",
        "scent_notes": ["green", "vegetable", "earthy", "herbal"],
        "category": "green",
        "description": "The signature green aroma molecule of bell peppers, cabernet sauvignon, and green peas."
    },
    {
        "name": "Furaneol (Strawberry Furanone)",
        "smiles": "CC1=C(O)C(=O)OC1C",
        "formula": "C6H8O3",
        "scent_notes": ["caramel", "sweet", "strawberry", "fruity"],
        "category": "sweet",
        "description": "Provides the sweet roasted caramel and wild strawberry aroma in ripe fruit and desserts."
    },
    {
        "name": "Allyl Hexanoate",
        "smiles": "CCCCC(=O)OCC=C",
        "formula": "C9H16O2",
        "scent_notes": ["fruity", "pineapple", "tropical", "sweet"],
        "category": "fruity",
        "description": "A staple ester replicating sweet, tropical pineapple and passion fruit bouquets."
    }
]


@app.on_event("startup")
def load_assets():
    """Load vocabulary, dataset stats, models, and thresholds at startup."""
    global VOCAB, DATASET_STATS, MODELS_CACHE, THRESHOLDS_CACHE
    
    # Load vocab
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            VOCAB = json.load(f)
        print(f"Loaded vocabulary: {len(VOCAB)} labels")
        
    # Load dataset stats
    if STATS_PATH.exists():
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            DATASET_STATS = json.load(f)
            
    # Load trained models
    num_classes = len(VOCAB) if VOCAB else 157
    for m_name in ["gatv2", "gcn"]:
        ckpt_path = MODEL_DIR / f"{m_name}_best.pt"
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                model = build_model(m_name, num_classes=num_classes).to(DEVICE)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                MODELS_CACHE[m_name] = model
                
                # Thresholds
                thresholds_list = ckpt.get("optimal_thresholds", [0.5] * num_classes)
                THRESHOLDS_CACHE[m_name] = {
                    VOCAB[i]: float(thresholds_list[i]) for i in range(len(VOCAB))
                }
                print(f"Loaded model '{m_name}' successfully.")
            except Exception as e:
                print(f"Error loading model {m_name}: {e}")


# Request & Response Schemas
class PredictRequest(BaseModel):
    smiles: str = Field(..., example="O=Cc1ccc(O)c(OC)c1", description="Valid molecular SMILES string")
    model: str = Field("gatv2", example="gatv2", description="Model architecture: 'gatv2' or 'gcn'")


class OdorPrediction(BaseModel):
    label: str
    confidence: float
    threshold: float
    is_positive: bool
    scent_family: str


class PredictResponse(BaseModel):
    smiles: str
    canonical_smiles: str
    model_used: str
    properties: Dict[str, Any]
    image_png: Optional[str]
    predictions: List[OdorPrediction]
    positive_labels: List[str]


class BatchPredictRequest(BaseModel):
    smiles_list: List[str] = Field(..., description="List of SMILES strings")
    model: str = Field("gatv2", description="Model architecture: 'gatv2' or 'gcn'")


from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request

FRONTEND_DIST = Path("frontend/dist")
if FRONTEND_DIST.exists() and (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


@app.get("/")
def health_or_index(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept and FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
        return FileResponse(FRONTEND_DIST / "index.html")
    return {
        "status": "online",
        "app": "ScentGraph",
        "description": "Molecule-to-Scent Predictor GNN",
        "available_models": list(MODELS_CACHE.keys()),
        "vocab_size": len(VOCAB)
    }


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "app": "ScentGraph",
        "available_models": list(MODELS_CACHE.keys()),
        "vocab_size": len(VOCAB)
    }


@app.post("/predict", response_model=PredictResponse)
def predict_molecule(req: PredictRequest):
    """Predicts odor descriptors and confidence scores for a single SMILES string."""
    smiles = req.smiles.strip()
    model_key = req.model.lower().strip()
    if model_key not in MODELS_CACHE:
        # Fallback to available model
        model_key = "gatv2" if "gatv2" in MODELS_CACHE else list(MODELS_CACHE.keys())[0]

    model = MODELS_CACHE.get(model_key)
    if model is None:
        raise HTTPException(status_code=503, detail="Models not loaded or trained yet.")

    # Featurize
    graph_data = smiles_to_graph(smiles)
    if graph_data is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or unparseable SMILES: '{smiles}'. Please verify molecular structure."
        )

    props = get_molecule_properties(smiles)
    img_b64 = render_molecule_png(smiles)

    # Model inference
    graph_data = graph_data.to(DEVICE)
    batch_idx = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        edge_attr = getattr(graph_data, 'edge_attr', None)
        logits = model(graph_data.x, graph_data.edge_index, batch_idx, edge_attr=edge_attr)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    thresholds_map = THRESHOLDS_CACHE.get(model_key, {})
    predictions: List[OdorPrediction] = []
    positive_labels: List[str] = []

    for i, label in enumerate(VOCAB):
        conf = round(float(probs[i]), 4)
        th = thresholds_map.get(label, 0.5)
        is_pos = bool(conf >= th)
        if is_pos:
            positive_labels.append(label)

        predictions.append(OdorPrediction(
            label=label,
            confidence=conf,
            threshold=round(th, 2),
            is_positive=is_pos,
            scent_family=get_scent_family(label)
        ))

    # Sort descending by confidence
    predictions.sort(key=lambda x: x.confidence, reverse=True)

    return PredictResponse(
        smiles=smiles,
        canonical_smiles=props["canonical_smiles"] if props else smiles,
        model_used=model_key,
        properties=props or {},
        image_png=img_b64,
        predictions=predictions,
        positive_labels=positive_labels
    )


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    """Batch inference endpoint for multiple SMILES strings."""
    results = []
    for s in req.smiles_list[:50]:  # Limit to 50 at a time
        try:
            res = predict_molecule(PredictRequest(smiles=s, model=req.model))
            results.append({"smiles": s, "success": True, "data": res})
        except Exception as e:
            results.append({"smiles": s, "success": False, "error": str(e)})
    return {"count": len(results), "results": results}


@app.get("/model/metrics")
def get_model_metrics(model: str = Query("gatv2")):
    """Returns training metrics and curves for active model."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found.")
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    m_key = model.lower()
    model_metrics = metrics_data.get("models", {}).get(m_key)
    if not model_metrics:
        m_key = list(metrics_data.get("models", {}).keys())[0]
        model_metrics = metrics_data["models"][m_key]

    return {
        "model": m_key,
        "metrics": model_metrics,
        "split_sizes": metrics_data.get("split_sizes"),
        "vocab_size": metrics_data.get("vocab_size")
    }


@app.get("/model/compare")
def get_model_comparison():
    """Returns side-by-side comparison of GCN vs GATv2."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found.")
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    gat_m = metrics_data.get("models", {}).get("gatv2", {})
    gcn_m = metrics_data.get("models", {}).get("gcn", {})

    # Compute comparative delta
    comparison_table = []
    for lbl in VOCAB:
        gat_f1 = gat_m.get("per_label_metrics", {}).get(lbl, {}).get("f1", 0.0)
        gcn_f1 = gcn_m.get("per_label_metrics", {}).get(lbl, {}).get("f1", 0.0)
        support = gat_m.get("per_label_metrics", {}).get(lbl, {}).get("support", 0)
        
        comparison_table.append({
            "label": lbl,
            "scent_family": get_scent_family(lbl),
            "gatv2_f1": gat_f1,
            "gcn_f1": gcn_f1,
            "gatv2_threshold": gat_m.get("optimal_thresholds", {}).get(lbl, 0.5),
            "gcn_threshold": gcn_m.get("optimal_thresholds", {}).get(lbl, 0.5),
            "support": support,
            "delta_f1": round(gat_f1 - gcn_f1, 4)
        })

    # Sort comparison table by support count descending
    comparison_table.sort(key=lambda x: x["support"], reverse=True)

    return {
        "timestamp": metrics_data.get("timestamp"),
        "total_molecules": metrics_data.get("total_molecules"),
        "vocab_size": metrics_data.get("vocab_size"),
        "gatv2_summary": {
            "display_name": gat_m.get("display_name", "ScentGATv2 (Attention)"),
            "macro_f1": gat_m.get("macro_f1"),
            "micro_f1": gat_m.get("micro_f1"),
            "test_loss": gat_m.get("test_loss"),
            "top_10_best": gat_m.get("top_10_best", []),
            "top_10_worst": gat_m.get("top_10_worst", [])
        },
        "gcn_summary": {
            "display_name": gcn_m.get("display_name", "GCN Baseline"),
            "macro_f1": gcn_m.get("macro_f1"),
            "micro_f1": gcn_m.get("micro_f1"),
            "test_loss": gcn_m.get("test_loss"),
            "top_10_best": gcn_m.get("top_10_best", []),
            "top_10_worst": gcn_m.get("top_10_worst", [])
        },
        "comparison_table": comparison_table,
        "gatv2_curves": gat_m.get("training_curves", []),
        "gcn_curves": gcn_m.get("training_curves", [])
    }


@app.get("/labels")
def get_labels():
    """Returns full vocabulary of odor labels with dataset frequencies and fragrance family."""
    freqs = DATASET_STATS.get("label_frequencies", {})
    results = []
    for lbl in VOCAB:
        count = freqs.get(lbl, 0)
        results.append({
            "label": lbl,
            "count": count,
            "percentage": round((count / max(1, DATASET_STATS.get("total_molecules", 1))) * 100, 2),
            "scent_family": get_scent_family(lbl)
        })
    results.sort(key=lambda x: x["count"], reverse=True)
    return {
        "total_labels": len(results),
        "total_molecules": DATASET_STATS.get("total_molecules", 0),
        "labels": results,
        "scent_families": list(SCENT_FAMILIES.keys())
    }


@app.get("/molecule/examples")
def get_molecule_examples():
    """Returns curated aroma molecules with pre-rendered structures and background info."""
    enriched = []
    for mol in CURATED_MOLECULES:
        enriched.append({
            **mol,
            "image_png": render_molecule_png(mol["smiles"], width=300, height=220)
        })
    return {"molecules": enriched}


# SPA Client-Side Routing Fallback
@app.get("/{full_path:path}", include_in_schema=False)
def serve_spa_route(full_path: str):
    if FRONTEND_DIST.exists():
        target = FRONTEND_DIST / full_path
        if full_path and target.exists() and target.is_file():
            return FileResponse(target)
        if (FRONTEND_DIST / "index.html").exists():
            return FileResponse(FRONTEND_DIST / "index.html")
    raise HTTPException(status_code=404, detail="Page not found")
