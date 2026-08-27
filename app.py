"""
ChurnIQ - Model serving API

Loads the trained model bundle (churniq_model.joblib, produced by
03_model_training.py) and exposes it over HTTP with FastAPI.

Endpoints:
  GET  /health          - liveness check + model metadata
  POST /predict          - churn probability + risk flag for one customer
  POST /predict/batch     - same, for a list of customers

Run locally:
  uvicorn app:app --reload --port 8000

Then see interactive docs at http://127.0.0.1:8000/docs
"""

import json
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "churniq_model.joblib"
METADATA_PATH = "churniq_model_metadata.json"

app = FastAPI(
    title="ChurnIQ Model API",
    description="Serves the trained ChurnIQ churn-prediction model for real-time scoring.",
    version="1.0.0",
)

# --- Load model bundle once at startup ---
try:
    bundle = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
except FileNotFoundError as e:
    raise RuntimeError(
        "Model file not found. Run the pipeline first: "
        "python3 01_generate_data.py && python3 02_feature_engineering.py && python3 03_model_training.py"
    ) from e

MODEL = bundle["model"]
SCALER = bundle["scaler"]
MODEL_TYPE = bundle["model_type"]
FEATURE_COLUMNS = bundle["feature_columns"]
THRESHOLD = bundle["threshold"]


# --- Request/response schemas ---
class CustomerFeatures(BaseModel):
    age: int = Field(..., example=42)
    average_watch_hours: float = Field(..., example=6.5)
    mobile_app_usage_pct: float = Field(..., example=35.0)
    complaints_raised: int = Field(..., example=1)
    received_promotions: int = Field(..., ge=0, le=1, example=0)
    referred_by_friend: int = Field(..., ge=0, le=1, example=0)
    monthly_fee: float = Field(..., example=13.99)
    tenure_days: int = Field(..., example=200)
    gender: str = Field(..., example="Female")
    country: str = Field(..., example="Ireland")
    subscription_type: str = Field(..., example="Standard")


class PredictionResponse(BaseModel):
    churn_probability: float
    predicted_churn: bool
    risk_tier: str
    model_type: str
    model_auc: float


def _featurize(customer: CustomerFeatures) -> pd.DataFrame:
    """Rebuild the exact engineered feature row the model was trained on."""
    row = {
        "age": customer.age,
        "average_watch_hours": customer.average_watch_hours,
        "mobile_app_usage_pct": customer.mobile_app_usage_pct,
        "complaints_raised": customer.complaints_raised,
        "received_promotions": customer.received_promotions,
        "referred_by_friend": customer.referred_by_friend,
        "monthly_fee": customer.monthly_fee,
        "tenure_days": customer.tenure_days,
        "watch_per_fee_ratio": customer.average_watch_hours / customer.monthly_fee,
        "is_heavy_complainer": int(customer.complaints_raised >= 2),
        "is_new_customer": int(customer.tenure_days < 90),
        "gender_Male": int(customer.gender == "Male"),
        "gender_Other": int(customer.gender == "Other"),
        "country_Canada": int(customer.country == "Canada"),
        "country_France": int(customer.country == "France"),
        "country_Germany": int(customer.country == "Germany"),
        "country_India": int(customer.country == "India"),
        "country_Ireland": int(customer.country == "Ireland"),
        "country_UK": int(customer.country == "UK"),
        "country_USA": int(customer.country == "USA"),
        "subscription_type_Premium": int(customer.subscription_type == "Premium"),
        "subscription_type_Standard": int(customer.subscription_type == "Standard"),
    }
    df = pd.DataFrame([row])
    # enforce the exact column order the model was trained on
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise HTTPException(status_code=500, detail=f"Feature mismatch, missing: {missing}")
    return df[FEATURE_COLUMNS]


def _risk_tier(p: float) -> str:
    if p >= 0.7:
        return "high"
    if p >= 0.4:
        return "medium"
    return "low"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "model_auc": metadata["auc"],
        "trained_on_rows": metadata["trained_on_rows"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    X = _featurize(customer)
    if MODEL_TYPE == "Logistic Regression":
        X_input = SCALER.transform(X)
    else:
        X_input = X
    proba = float(MODEL.predict_proba(X_input)[0, 1])
    return PredictionResponse(
        churn_probability=round(proba, 4),
        predicted_churn=proba >= THRESHOLD,
        risk_tier=_risk_tier(proba),
        model_type=MODEL_TYPE,
        model_auc=metadata["auc"],
    )


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(customers: List[CustomerFeatures]):
    return [predict(c) for c in customers]


@app.get("/")
def root():
    return {
        "message": "ChurnIQ model API. See /docs for interactive documentation.",
        "endpoints": ["/health", "/predict", "/predict/batch"],
    }
