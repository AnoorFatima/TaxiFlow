from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "q_learning_model.npz"
DATA_PATH = BASE_DIR / "data" / "demand_by_zone_15min.csv"
METRICS_PATH = BASE_DIR / "performance_analysis.csv"


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="TaxiFlow API",
    description="Taxi Fleet Optimization Using Reinforcement Learning",
    version="1.0.0",
)


# --------------------------------------------------
# Global variables
# --------------------------------------------------

q_table: dict[tuple, Any] = {}
demand_df = pd.DataFrame()
metrics_df = pd.DataFrame()

model_loaded = False
model_message = "Model has not been loaded."


# --------------------------------------------------
# Convert lists and NumPy values into hashable keys
# --------------------------------------------------

def make_hashable(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return make_hashable(value.item())
        return tuple(make_hashable(item) for item in value.tolist())

    if isinstance(value, (list, tuple)):
        return tuple(make_hashable(item) for item in value)

    if isinstance(value, np.generic):
        return value.item()

    return value


# --------------------------------------------------
# Load Q-learning model
# --------------------------------------------------

def load_q_learning_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Q-learning model not found: {MODEL_PATH}"
        )

    loaded_table = {}

    with np.load(MODEL_PATH, allow_pickle=True) as model:
        model_keys = model.files

        if "q_table" in model_keys:
            raw_table = model["q_table"]

            if isinstance(raw_table, np.ndarray) and raw_table.ndim == 0:
                raw_table = raw_table.item()

            elif (
                isinstance(raw_table, np.ndarray)
                and raw_table.dtype == object
                and raw_table.size == 1
                and isinstance(raw_table.flat[0], dict)
            ):
                raw_table = raw_table.flat[0]

            if not isinstance(raw_table, dict):
                raise ValueError(
                    "The saved q_table is not a dictionary. "
                    f"Available keys: {model_keys}"
                )

            for state, values in raw_table.items():
                state_key = make_hashable(state)
                loaded_table[state_key] = np.asarray(values).reshape(-1)

        elif "states" in model_keys and "q_values" in model_keys:
            states = model["states"]
            q_values = model["q_values"]

            if len(states) != len(q_values):
                raise ValueError(
                    "Saved states and Q-values have different lengths."
                )

            for state, values in zip(states, q_values):
                state_key = make_hashable(state)
                loaded_table[state_key] = np.asarray(values).reshape(-1)

        else:
            raise ValueError(
                "Unsupported Q-learning model format. "
                f"Available keys: {model_keys}"
            )

    if not loaded_table:
        raise ValueError("The saved Q-table is empty.")

    return loaded_table


# --------------------------------------------------
# Load project files
# --------------------------------------------------

try:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Demand dataset not found: {DATA_PATH}"
        )

    demand_df = pd.read_csv(DATA_PATH)

    if "zone_id" not in demand_df.columns:
        raise ValueError(
            "The demand dataset must contain a zone_id column."
        )

    if MODEL_PATH.exists():
        q_table = load_q_learning_model()
        model_loaded = True
        model_message = (
            f"Q-learning model loaded successfully. "
            f"Learned states: {len(q_table)}"
        )
    else:
        model_message = (
            "Demand data loaded, but q_learning_model.npz was not found."
        )

    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)

except Exception as exc:
    raise RuntimeError(
        f"Could not load TaxiFlow project files: {exc}"
    ) from exc


# --------------------------------------------------
# Request model
# --------------------------------------------------

class RecommendationRequest(BaseModel):
    demand_level: int = Field(ge=0, le=3)
    coverage_level: int = Field(ge=0, le=2)
    time_period: int = Field(ge=0, le=3)


# --------------------------------------------------
# Action descriptions
# --------------------------------------------------

ACTION_DESCRIPTIONS = {
    0: "Keep the current taxi allocation.",
    1: (
        "Recommend moving a taxi from a low-demand zone "
        "to the highest-demand zone."
    ),
    2: (
        "Recommend moving a taxi from a low demand-per-taxi zone "
        "to the zone with the highest demand per taxi."
    ),
}


# --------------------------------------------------
# API routes
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "TaxiFlow",
        "description": "Taxi Fleet Optimization Using Reinforcement Learning",
        "status": "running",
        "model_loaded": model_loaded,
        "model_message": model_message,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "demand_data_loaded": not demand_df.empty,
    }


@app.get("/zones")
def get_zones():
    zones = (
        demand_df["zone_id"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return {
        "total_zones": len(zones),
        "zones": zones,
    }


@app.get("/metrics")
def get_metrics():
    if metrics_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "Performance metrics were not found. "
                "Run analyze_results.py first."
            ),
        )

    return {
        "total_policies": len(metrics_df),
        "metrics": metrics_df.to_dict(orient="records"),
    }


@app.post("/recommend")
def recommend_action(request: RecommendationRequest):
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="The Q-learning model is not loaded.",
        )

    state = (
        request.demand_level,
        request.coverage_level,
        request.time_period,
    )

    state_key = make_hashable(state)

    if state_key not in q_table:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "State not found in the learned Q-table.",
                "requested_state": list(state),
                "available_learned_states": len(q_table),
            },
        )

    q_values = np.asarray(q_table[state_key]).reshape(-1)

    if q_values.size == 0:
        raise HTTPException(
            status_code=500,
            detail="The selected state has no Q-values.",
        )

    action = int(np.argmax(q_values))

    return {
        "state": {
            "demand_level": request.demand_level,
            "coverage_level": request.coverage_level,
            "time_period": request.time_period,
        },
        "recommended_action": action,
        "recommendation": ACTION_DESCRIPTIONS.get(
            action,
            "The model selected an unknown action.",
        ),
        "q_values": q_values.tolist(),
        "note": (
            "This recommendation comes from the trained simulation model, "
            "not a live taxi dispatch system."
        ),
    }