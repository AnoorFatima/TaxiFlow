
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "q_learning_model.npz"
DATA_PATH = BASE_DIR / "data" / "demand_by_zone_15min.csv"
METRICS_PATH = BASE_DIR / "performance_analysis.csv"

app = FastAPI(
    title="TaxiFlow API",
    description="Taxi Fleet Optimization Using Reinforcement Learning",
    version="1.0.0"
)

# Load demand data
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Demand data not found: {DATA_PATH}")

demand_df = pd.read_csv(DATA_PATH)

# Load the trained Q-learning model
q_table = {}

if MODEL_PATH.exists():
    model = np.load(MODEL_PATH)

    for state, values in zip(model["states"], model["q_values"]):
        q_table[tuple(state.tolist())] = values

# Load performance metrics if available
if METRICS_PATH.exists():
    metrics_df = pd.read_csv(METRICS_PATH)
else:
    metrics_df = pd.DataFrame()

# Request format
class RecommendationRequest(BaseModel):
    demand_level: int = Field(ge=0, le=3)
    coverage_level: int = Field(ge=0, le=2)
    time_period: int = Field(ge=0, le=3)


# Action descriptions
ACTIONS = {
    0: "Keep the current taxi allocation.",
    1: "Move a taxi toward the highest-demand zone.",
    2: "Move a taxi toward the highest demand-per-taxi zone."
}


# 1. Home endpoint
@app.get("/")
def home():
    return {
        "project": "TaxiFlow",
        "status": "running",
        "model_loaded": bool(q_table)
    }


# 2. Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": bool(q_table),
        "demand_data_loaded": not demand_df.empty
    }


# 3. List taxi zones
@app.get("/zones")
def get_zones():
    zones = sorted(demand_df["zone_id"].dropna().unique().tolist())

    return {
        "total_zones": len(zones),
        "zones": zones
    }


# 4. Show performance metrics
@app.get("/metrics")
def get_metrics():
    if metrics_df.empty:
        raise HTTPException(
            status_code=404,
            detail="Performance metrics file is unavailable."
        )

    return metrics_df.to_dict(orient="records")


# 5. Recommend an action
@app.post("/recommend")
def recommend(request: RecommendationRequest):
    if not q_table:
        raise HTTPException(
            status_code=503,
            detail="Q-learning model is not loaded."
        )

    state = (
        request.demand_level,
        request.coverage_level,
        request.time_period
    )

    if state not in q_table:
        raise HTTPException(
            status_code=404,
            detail="This state was not learned by the model."
        )

    q_values = q_table[state]
    action = int(np.argmax(q_values))

    return {
        "state": state,
        "recommended_action": action,
        "recommendation": ACTIONS[action],
        "q_values": q_values.tolist(),
        "note": "Simulation-based recommendation, not live taxi dispatch."
    }