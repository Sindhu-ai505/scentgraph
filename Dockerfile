# Multi-stage Dockerfile for ScentGraph Web Deployment

# Stage 1: Build React Frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend & Serving Environment
FROM python:3.12-slim

# Install system dependencies for RDKit and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code, models, and processed data
COPY backend/ backend/
COPY data/processed/ data/processed/

# Copy built frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist frontend/dist

# Expose port (default 8080)
EXPOSE 8080
ENV PORT=8080
ENV HOST=0.0.0.0

# Start unified FastAPI server serving both API and Frontend
CMD ["sh", "-c", "python -m uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
