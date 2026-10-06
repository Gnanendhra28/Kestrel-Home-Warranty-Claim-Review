"""
FastAPI Single-Record Prediction Service for Kestrel Home Warranty Fraud Review.

Endpoints:
- GET /health: Health check, model status, feature count.
- GET /examples: Pre-configured reference claims for quick testing across risk tiers.
- POST /predict: Scores a single incoming claim before payout, computes expected fraud value,
  determines review recommendation, and outputs deterministic human-readable reasons.
"""

from datetime import datetime
import json
import os
from typing import Dict, List, Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.features.build_features import (
    load_csv,
    TemporalHistoryIndex,
    build_feature_dict,
)
from src.business.risk_engine import (
    REVIEW_COST_INR,
    DISCLAIMER_TEXT,
    compute_expected_fraud_value,
    determine_risk_level,
    determine_review_recommendation,
    generate_risk_reasons,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

app = FastAPI(
    title="Kestrel Warranty Fraud Review API",
    description="Operational single-record fraud risk scoring and decision support for warranty claims prior to payout.",
    version="1.0.0",
)

# Global runtime state
MODEL_PIPELINE = None
MODEL_FEATURES = []
PRODUCTS_MAP: Dict[str, Dict[str, str]] = {}
PARTNERS_MAP: Dict[str, Dict[str, str]] = {}
HISTORY_INDEX: Optional[TemporalHistoryIndex] = None


@app.on_event("startup")
def startup_load_artifacts():
    global MODEL_PIPELINE, MODEL_FEATURES, PRODUCTS_MAP, PARTNERS_MAP, HISTORY_INDEX

    # 1. Feature list
    feat_spec_path = os.path.join(MODELS_DIR, "feature_list.json")
    with open(feat_spec_path, "r", encoding="utf-8") as f:
        feat_spec = json.load(f)
    MODEL_FEATURES = feat_spec["features"]

    # 2. Model pipeline
    model_path = os.path.join(MODELS_DIR, "model_pipeline.joblib")
    MODEL_PIPELINE = joblib.load(model_path)

    # 3. Master reference catalogs
    _, partners_raw = load_csv("partners.csv")
    PARTNERS_MAP = {r["partner_id"]: r for r in partners_raw}

    _, prods_raw = load_csv("products.csv")
    PRODUCTS_MAP = {r["sku"]: r for r in prods_raw}

    # 4. Historical index for strictly prior temporal features
    _, train_raw = load_csv("train.csv")
    for r in train_raw:
        r["submitted_dt"] = datetime.fromisoformat(r["submitted_at"])

    history_idx = TemporalHistoryIndex()
    history_idx.populate(train_raw, label_cutoff=datetime(2026, 5, 1))
    HISTORY_INDEX = history_idx


class ClaimRequest(BaseModel):
    claim_id: str = Field(default="WC-INCOMING-001", description="Unique warranty claim ID")
    submitted_at: str = Field(default="2026-07-15T11:00:00", description="ISO 8601 submission timestamp")
    partner_id: str = Field(..., description="Service partner ID (e.g. SP3033)")
    sku: str = Field(..., description="Product SKU code (e.g. KH-AF-01)")
    product_serial: str = Field(..., description="Serial number of the serviced appliance")
    days_since_purchase: float = Field(..., ge=0, description="Elapsed days between appliance purchase and claim filing")
    claim_amount_inr: float = Field(..., gt=0, description="Claimed repair/reimbursement amount in INR")
    photo_attached: str = Field(default="Y", description="Photographic proof attached ('Y' or 'N')")
    customer_prior_claims: int = Field(default=0, ge=0, description="Number of prior claims by this customer")
    claim_description: str = Field(default="unit not heating", description="Partner-provided fault description")

    @field_validator("photo_attached")
    @classmethod
    def validate_photo_flag(cls, v: str) -> str:
        clean = v.strip().upper()
        if clean not in ("Y", "N"):
            raise ValueError("photo_attached must be 'Y' or 'N'")
        return clean

    @field_validator("submitted_at")
    @classmethod
    def validate_iso_timestamp(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except Exception as e:
            raise ValueError(f"submitted_at must be valid ISO 8601 datetime: {e}")
        return v


class ClaimPredictionResponse(BaseModel):
    claim_id: str
    fraud_risk_score: float
    risk_level: str
    claim_amount_inr: float
    estimated_expected_fraud_value_inr: float
    review_cost_inr: float
    review_recommended: bool
    reasons: List[str]
    disclaimer: str


@app.get("/health", tags=["Monitoring"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_PIPELINE is not None,
        "features_count": len(MODEL_FEATURES),
        "partners_indexed": len(PARTNERS_MAP),
        "products_indexed": len(PRODUCTS_MAP),
    }


@app.get("/examples", tags=["Reference Data"])
def get_sample_claims():
    """
    Returns verified archetypal claims across High, Medium, and Low risk profiles
    for convenient manual review testing.
    """
    return [
        {
            "label": "High Risk — Recycled Serial & High Value",
            "claim": {
                "claim_id": "DEMO-HIGH-01",
                "submitted_at": "2026-07-20T14:30:00",
                "partner_id": "SP3033",
                "sku": "KH-AF-01",
                "product_serial": "AF-2023-8812",
                "days_since_purchase": 350.0,
                "claim_amount_inr": 4800.0,
                "photo_attached": "N",
                "customer_prior_claims": 3,
                "claim_description": "unit not heating tripping mcb",
            },
        },
        {
            "label": "Medium Risk — Sub-₹2,000 Threshold Bunching",
            "claim": {
                "claim_id": "DEMO-MED-02",
                "submitted_at": "2026-07-18T10:15:00",
                "partner_id": "SP3032",
                "sku": "KH-IC-01",
                "product_serial": "IC-2024-4419",
                "days_since_purchase": 180.0,
                "claim_amount_inr": 1950.0,
                "photo_attached": "N",
                "customer_prior_claims": 1,
                "claim_description": "power button not working",
            },
        },
        {
            "label": "Lower Risk — Routine Minor Repair with Photo",
            "claim": {
                "claim_id": "DEMO-LOW-03",
                "submitted_at": "2026-07-16T16:00:00",
                "partner_id": "SP3265",
                "sku": "KH-CF-01",
                "product_serial": "CF-2024-0012",
                "days_since_purchase": 90.0,
                "claim_amount_inr": 850.0,
                "photo_attached": "Y",
                "customer_prior_claims": 0,
                "claim_description": "blade jammed loud noise while running",
            },
        },
    ]


@app.post("/predict", response_model=ClaimPredictionResponse, tags=["Scoring"])
def predict_claim(req: ClaimRequest):
    if MODEL_PIPELINE is None or HISTORY_INDEX is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts or historical index not yet initialized.",
        )

    # 1. Validate SKU
    if req.sku not in PRODUCTS_MAP:
        valid_skus = list(PRODUCTS_MAP.keys())
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown SKU '{req.sku}'. Must be one of: {valid_skus}",
        )

    # 2. Validate or fallback Partner ID
    if req.partner_id not in PARTNERS_MAP:
        valid_partners = list(PARTNERS_MAP.keys())[:10]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown partner_id '{req.partner_id}'. Examples of valid IDs: {valid_partners}",
        )

    submitted_dt = datetime.fromisoformat(req.submitted_at)

    raw_row = {
        "claim_id": req.claim_id,
        "submitted_at": req.submitted_at,
        "submitted_dt": submitted_dt,
        "partner_id": req.partner_id,
        "sku": req.sku,
        "product_serial": req.product_serial,
        "days_since_purchase": req.days_since_purchase,
        "claim_amount_inr": req.claim_amount_inr,
        "photo_attached": req.photo_attached,
        "customer_prior_claims": req.customer_prior_claims,
        "claim_description": req.claim_description,
    }

    # 3. Extract production feature vector
    feat_dict = build_feature_dict(raw_row, PRODUCTS_MAP, PARTNERS_MAP, HISTORY_INDEX)

    # 4. Construct DataFrame matching model features
    X_single = pd.DataFrame([feat_dict])[MODEL_FEATURES]

    # 5. Predict continuous probability
    score = float(MODEL_PIPELINE.predict_proba(X_single)[:, 1][0])
    score = round(score, 6)

    # 6. Business evaluation
    expected_fraud_val = compute_expected_fraud_value(score, req.claim_amount_inr)
    risk_level = determine_risk_level(score)
    review_rec = determine_review_recommendation(expected_fraud_val, REVIEW_COST_INR)
    reasons = generate_risk_reasons(feat_dict, score, expected_fraud_val)

    return ClaimPredictionResponse(
        claim_id=req.claim_id,
        fraud_risk_score=score,
        risk_level=risk_level,
        claim_amount_inr=req.claim_amount_inr,
        estimated_expected_fraud_value_inr=expected_fraud_val,
        review_cost_inr=REVIEW_COST_INR,
        review_recommended=review_rec,
        reasons=reasons,
        disclaimer=DISCLAIMER_TEXT,
    )


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.api.main:app", host=host, port=port)
