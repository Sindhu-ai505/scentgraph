---
title: ScentGraph — Molecule-to-Scent GNN
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8080
pinned: false
license: mit
---

# ScentGraph — Molecule-to-Scent Predictor GNN

**ScentGraph** is a full-stack deep learning web application that predicts the odor descriptors and olfactory bouquet of chemical molecules directly from their SMILES representation. Powered by **Graph Neural Networks (PyTorch Geometric)** and **RDKit**, ScentGraph converts molecular graphs into high-dimensional scent profiles across 157 distinct fragrance categories.

---

## Architecture Overview

```
ScentGraph/
├── backend/
│   ├── api.py               # FastAPI server with prediction & analytics endpoints
│   ├── features.py          # RDKit molecular graph featurization (38 node, 7 edge features)
│   ├── model.py             # ScentGATv2 (Graph Attention) & GCNBaseline models
│   ├── train.py             # Stratified training pipeline with per-label threshold optimization
│   ├── models/              # Checkpoints (gatv2_best.pt, gcn_best.pt, metrics.json)
│   ├── tests/               # PyTest test suite (features, models, API)
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/      # Navbar, HowItWorksModal, etc.
│   │   ├── pages/           # PredictorScreen, ModelCompareScreen, DatasetExplorer, TrainingDashboard
│   │   ├── services/        # api.js client
│   │   ├── utils/           # scentColors.js fragrance palette tokens
│   │   ├── App.jsx          # Main application layout
│   │   └── index.css        # Tailwind styling with glassmorphism & scent glow effects
│   ├── package.json         # React + Vite + Tailwind + Framer Motion + Recharts
│   └── vite.config.js       # Vite configuration
├── data/
│   ├── build_dataset.py     # End-to-end fetch & clean pipeline for Pyrfume archive
│   ├── raw/                 # Cached raw CSVs from leffingwell, goodscents, flavornet, aromadb
│   └── processed/           # molecules_odors.parquet, molecules_odors.csv, stats & vocab
├── .env.example             # Configurable ports & paths
└── README.md
```

---

## 1. Dataset & Sensory Data Provenance

The dataset was fetched and merged from 4 peer-reviewed public datasets in the **Pyrfume-data** archive:
1. **Leffingwell Odor Dataset** (~3,500 molecules)
2. **GoodScents Perfumery Database** (~4,600 molecules)
3. **Flavornet** (~700 molecules)
4. **AromaDB** (~800 molecules)

### Processing & Cleaning
- Molecules parsed and canonicalized using RDKit (`Chem.MolToSmiles(canonical=True)`).
- 15 unparseable/invalid SMILES dropped.
- Multi-source odor descriptors normalized and deduplicated per canonical molecule.
- Filtered out sparse labels appearing fewer than 30 times.
- **Final Dataset**: **5,070 unique molecules** across **157 validated odor descriptors** (average 4.89 descriptors/molecule).

To re-run the data pipeline from scratch:
```bash
python data/build_dataset.py
```

---

## 2. Graph Neural Network Modeling

### Featurization (`backend/features.py`)
- **Nodes (Atoms)**: 38-dimensional feature vector:
  - Atomic number (C, N, O, F, P, S, Cl, Br, I, other)
  - Degree (0–5)
  - Formal charge (-2 to +2)
  - Hybridization ($sp, sp^2, sp^3, sp^3d, sp^3d^2$)
  - Aromaticity (binary)
  - Ring membership (binary)
  - Total hydrogen count (0–4)
- **Edges (Bonds)**: 7-dimensional feature vector:
  - Bond type (SINGLE, DOUBLE, TRIPLE, AROMATIC)
  - Conjugation state (binary)
  - Ring membership (binary)

### Models
1. **ScentGATv2 (Champion)**:
   - 4 rounds of edge-conditioned `GATv2Conv` with multi-head attention.
   - Learned **Global Attention Pooling** gate to dynamically weight scent-critical osmophores.
   - Residual connections and LayerNorm.
2. **GCN Baseline**:
   - 3 rounds of `GCNConv` with concatenated Mean & Max pooling.

### Training & Per-Label Threshold Tuning (`backend/train.py`)
- **Loss**: Binary Cross-Entropy with positive class balancing (`pos_weight = (num_neg / (num_pos + 1e-5))`, clipped to `[1.0, 25.0]`).
- **Splits**: Multi-label iterative stratified splitting (80% Train / 10% Val / 10% Test).
- **Threshold Optimization**: Rather than a fixed 0.5 probability cutoff, optimal decision thresholds $t \in [0.10, 0.85]$ were tuned per-label on the validation set to maximize individual F1 scores.

To re-train the models:
```bash
python -m backend.train
```

---

## 3. Empirical Test Results

| Metric | ScentGATv2 (Attention) | GCN Baseline |
| :--- | :--- | :--- |
| **Test Macro-F1** | **0.2169** | 0.2077 |
| **Test Micro-F1** | 0.3242 | **0.3311** |
| **Test BCE Loss** | **0.2803** | 0.2847 |
| **Best Val Macro-F1** | 0.1748 | 0.1784 |
| **Top Predicted Accords** | `sulfur` (0.71), `woody` (0.71), `fruity` (0.69), `meaty` (0.64), `roasted` (0.61) | `sulfur` (0.76), `woody` (0.72), `fruity` (0.69), `meaty` (0.65), `roasted` (0.63) |

---

## 4. Running Locally

### Backend Setup
```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Start FastAPI server
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at `http://127.0.0.1:8000/docs`.

### Frontend Setup
```bash
# 1. Navigate to frontend
cd frontend

# 2. Install npm packages
npm install

# 3. Start Vite dev server
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 5. Automated Unit Tests
Run the test suite covering featurization, graph conversion, model forward passes, and API endpoints:
```bash
python -m pytest backend/tests -v
```
All 17 tests pass with full coverage.

---

## 6. Web Deployment (Free Cloud Hosting)

ScentGraph is configured for unified single-container cloud web deployment serving both the FastAPI REST API and React frontend on a single port.

### Option A: Hugging Face Spaces (Recommended - Free CPU/GPU)
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set Space Name: `scentgraph`, License: `MIT`.
3. Select Space SDK: **Docker** -> **Blank**.
4. Push your repository to the Hugging Face Space git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/scentgraph
   git push hf main
   ```
5. Hugging Face will automatically build the `Dockerfile` and launch the web app with a permanent public HTTPS URL!

### Option B: Render (Free Cloud Web Service)
1. Push this repository to GitHub.
2. Sign in to [dashboard.render.com](https://dashboard.render.com) and click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Render will detect `render.yaml` and the `Dockerfile` automatically.
5. Click **Create Web Service**. Your app will be live at `https://scentgraph.onrender.com`.

### Option C: Local / VPS Docker
```bash
# Build and run container locally
docker-compose up --build
```
Access the application at `http://localhost:8080`.
