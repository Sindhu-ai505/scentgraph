/**
 * ScentGraph API Service
 */

const API_BASE = import.meta.env.VITE_API_URL !== undefined
  ? import.meta.env.VITE_API_URL
  : (typeof window !== "undefined" && window.location.port === "5173" ? "http://127.0.0.1:8000" : "");

export async function predictMolecule(smiles, model = "gatv2") {
  const resp = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles, model }),
  });
  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed (${resp.status})`);
  }
  return resp.json();
}

export async function getModelComparison() {
  const resp = await fetch(`${API_BASE}/model/compare`);
  if (!resp.ok) {
    throw new Error(`Failed to load model comparison (${resp.status})`);
  }
  return resp.json();
}

export async function getModelMetrics(model = "gatv2") {
  const resp = await fetch(`${API_BASE}/model/metrics?model=${model}`);
  if (!resp.ok) {
    throw new Error(`Failed to load metrics (${resp.status})`);
  }
  return resp.json();
}

export async function getLabels() {
  const resp = await fetch(`${API_BASE}/labels`);
  if (!resp.ok) {
    throw new Error(`Failed to load label vocabulary (${resp.status})`);
  }
  return resp.json();
}

export async function getMoleculeExamples() {
  const resp = await fetch(`${API_BASE}/molecule/examples`);
  if (!resp.ok) {
    throw new Error(`Failed to load example molecules (${resp.status})`);
  }
  return resp.json();
}
