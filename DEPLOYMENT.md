# ScentGraph Web Deployment Guide

This guide walks you through deploying **ScentGraph** as a live web application on **Hugging Face Spaces** or **Render** (both free).

---

## Architecture of the Deployment

ScentGraph uses a **unified single-container design**:
- The React frontend is pre-built into production assets (`frontend/dist`).
- The FastAPI backend serves both the REST API endpoints (`/predict`, `/labels`, `/model/compare`) and the React web application on the **same port** (`8080`).
- No CORS issues, no separate hosting services required.

---

## Method 1: Hugging Face Spaces (Recommended — Free & Easiest for ML/Chemistry)

Hugging Face Spaces provides free 2 vCPU + 16 GB RAM hosting, which is ideal for PyTorch Geometric and RDKit.

### Steps:
1. **Create an account / Sign in** at [huggingface.co](https://huggingface.co).
2. Click on your profile icon in the top right → **New Space** (or navigate to [huggingface.co/new-space](https://huggingface.co/new-space)).
3. Fill in the fields:
   - **Space name**: `scentgraph`
   - **License**: `mit`
   - **Select the Space SDK**: Choose **Docker** → **Blank**
   - **Space hardware**: Select **CPU basic (free)**
4. Click **Create Space**.
5. Push this codebase to your new Space:
   Open a terminal in `ScentGraph`:
   ```bash
   # Initialize git if you haven't already
   git init
   git add .
   git commit -m "Deploy ScentGraph web application"

   # Add your Hugging Face space remote
   git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/scentgraph

   # Push to deploy
   git push hf main --force
   ```
6. Hugging Face will automatically trigger the Docker build. Within 3–4 minutes, your live web app will be accessible at:
   `https://huggingface.co/spaces/<YOUR_HF_USERNAME>/scentgraph`

---

## Method 2: Render (Free Cloud Web Service)

Render provides free Docker web service hosting connected directly to GitHub.

### Steps:
1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial ScentGraph commit"
   # Create a repository on github.com, then:
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/ScentGraph.git
   git branch -M main
   git push -u origin main
   ```
2. Go to [dashboard.render.com](https://dashboard.render.com) and click **New +** → **Web Service**.
3. Select **Build and deploy from a Git repository**.
4. Connect your `ScentGraph` GitHub repo.
5. Render will automatically read `render.yaml` and `Dockerfile`:
   - **Name**: `scentgraph`
   - **Runtime**: `Docker`
   - **Instance Type**: `Free`
6. Click **Create Web Service**.
7. Render will build the image and assign a public HTTPS URL (e.g., `https://scentgraph.onrender.com`).

---

## Method 3: Local or VPS Docker Deployment

To run the container on your own machine or a remote VPS:

```bash
# 1. Build and run using Docker Compose
docker-compose up --build -d

# 2. Verify container is running
docker ps
```
Open `http://localhost:8080` in any browser.

---

## Environment Variables Reference

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PORT` | `8080` | Port for the unified web server |
| `HOST` | `0.0.0.0` | Bind address |
| `DATA_PATH` | `data/processed/molecules_odors.parquet` | Path to the cleaned odor dataset |
| `MODEL_DIR` | `backend/models` | Path to trained model checkpoints |
