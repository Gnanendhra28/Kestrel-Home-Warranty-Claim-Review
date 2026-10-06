"""
Phase 3 Verification Tests for Kestrel Home Warranty Claim Review.

Tests:
1. predictions.csv format, row count (2,252), columns, exact claim_id order, score bounds.
2. Expected value engine calculation and review recommendation boundary.
3. Deterministic human-readable explanations (3-5 reasons, zero 'confirmed fraud' claims).
4. FastAPI health and single-claim prediction endpoints (HTTP 200).
5. FastAPI validation error handling on invalid SKU and malformed inputs (HTTP 422).
"""

import json
import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.business.risk_engine import (
    REVIEW_COST_INR,
    compute_expected_fraud_value,
    determine_risk_level,
    determine_review_recommendation,
    generate_risk_reasons,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def test_predictions_csv_integrity():
    preds_path = os.path.join(OUTPUTS_DIR, "predictions.csv")
    sample_sub_path = os.path.join(RAW_DATA_DIR, "sample_submission.csv")

    assert os.path.exists(preds_path), "outputs/predictions.csv does not exist!"

    df_preds = pd.read_csv(preds_path)
    df_sample = pd.read_csv(sample_sub_path)

    # 1. Exact 2,252 rows
    assert len(df_preds) == 2252, f"Expected 2252 rows, found {len(df_preds)}"

    # 2. Exact columns
    assert list(df_preds.columns) == ["claim_id", "score"], f"Columns mismatch: {list(df_preds.columns)}"

    # 3. Exact claim_id order
    assert (df_preds["claim_id"].values == df_sample["claim_id"].values).all(), (
        "claim_id column does not match sample_submission.csv order!"
    )

    # 4. No NaNs or infinities
    assert df_preds["score"].isna().sum() == 0, "Null or NaN values in scores"

    # 5. Scores between 0.0 and 1.0
    assert (df_preds["score"] >= 0.0).all() and (df_preds["score"] <= 1.0).all(), (
        "Scores contain values outside [0, 1]"
    )

    # 6. Continuous distribution
    assert df_preds["score"].nunique() > 100, "Scores should be continuous ranking probabilities"


def test_expected_value_and_risk_tiers():
    # Test arithmetic
    assert compute_expected_fraud_value(0.10, 2000.0) == 200.0
    assert compute_expected_fraud_value(0.50, 1000.0) == 500.0

    # Test review recommendation (cutoff 380)
    assert determine_review_recommendation(380.01) is True
    assert determine_review_recommendation(380.00) is False
    assert determine_review_recommendation(150.00) is False

    # Test risk tiers
    assert determine_risk_level(0.45) == "HIGH"
    assert determine_risk_level(0.40) == "HIGH"
    assert determine_risk_level(0.25) == "MEDIUM"
    assert determine_risk_level(0.20) == "MEDIUM"
    assert determine_risk_level(0.15) == "LOWER"


def test_human_readable_reasons_guardrails():
    dummy_feat = {
        "claim_amount_inr": 4800.0,
        "product_list_price": 5199.0,
        "claim_amount_to_list_price_ratio": 0.923,
        "serial_prior_claim_count": 2,
        "partner_prior_fraud_count": 1,
        "partner_smoothed_fraud_rate": 0.035,
        "partner_claims_prev_7d": 10,
        "partner_claim_velocity_ratio": 2.1,
        "photo_attached": 0,
        "customer_prior_claims": 3,
        "is_sub_2000": 0,
        "days_since_purchase": 340,
        "is_near_warranty_expiry": 1,
        "product_warranty_months": 12,
    }

    score = 0.42
    exp_val = compute_expected_fraud_value(score, dummy_feat["claim_amount_inr"])
    reasons = generate_risk_reasons(dummy_feat, score, exp_val)

    # 1. 3 to 5 reasons
    assert 3 <= len(reasons) <= 5, f"Expected 3-5 reasons, got {len(reasons)}"

    # 2. Never claim 'confirmed fraud' for the current claim
    for r in reasons:
        assert "this claim is confirmed fraud" not in r.lower()
        assert "fraudulent claim confirmed" not in r.lower()

    # 3. Serial recycling reason should be triggered
    assert any("serial" in r.lower() for r in reasons)


def test_api_health_endpoint():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["features_count"] == 33


def test_api_predict_valid_claim():
    with TestClient(app) as client:
        payload = {
            "claim_id": "TEST-CLAIM-001",
            "submitted_at": "2026-07-20T14:30:00",
            "partner_id": "SP3033",
            "sku": "KH-AF-01",
            "product_serial": "AF-2023-8812",
            "days_since_purchase": 350.0,
            "claim_amount_inr": 4800.0,
            "photo_attached": "N",
            "customer_prior_claims": 3,
            "claim_description": "unit not heating tripping mcb",
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["claim_id"] == "TEST-CLAIM-001"
        assert 0.0 <= data["fraud_risk_score"] <= 1.0
        assert data["risk_level"] in ("HIGH", "MEDIUM", "LOWER")
        assert data["claim_amount_inr"] == 4800.0
        assert data["review_cost_inr"] == REVIEW_COST_INR
        assert data["estimated_expected_fraud_value_inr"] == round(
            data["fraud_risk_score"] * 4800.0, 2
        )
        assert data["review_recommended"] == (data["estimated_expected_fraud_value_inr"] > REVIEW_COST_INR)
        assert 3 <= len(data["reasons"]) <= 5
        assert "DISCLAIMER" in data["disclaimer"]


def test_api_predict_invalid_inputs():
    with TestClient(app) as client:
        # 1. Invalid SKU
        res = client.post("/predict", json={
            "claim_id": "ERR-01",
            "submitted_at": "2026-07-20T14:30:00",
            "partner_id": "SP3033",
            "sku": "INVALID-SKU-999",
            "product_serial": "AF-2023-8812",
            "days_since_purchase": 100.0,
            "claim_amount_inr": 2000.0,
            "photo_attached": "Y",
            "customer_prior_claims": 0,
            "claim_description": "unit not heating",
        })
        assert res.status_code == 422

        # 2. Invalid photo flag
        res = client.post("/predict", json={
            "claim_id": "ERR-02",
            "submitted_at": "2026-07-20T14:30:00",
            "partner_id": "SP3033",
            "sku": "KH-AF-01",
            "product_serial": "AF-2023-8812",
            "days_since_purchase": 100.0,
            "claim_amount_inr": 2000.0,
            "photo_attached": "MAYBE",
            "customer_prior_claims": 0,
            "claim_description": "unit not heating",
        })
        assert res.status_code == 422

        # 3. Negative claim amount
        res = client.post("/predict", json={
            "claim_id": "ERR-03",
            "submitted_at": "2026-07-20T14:30:00",
            "partner_id": "SP3033",
            "sku": "KH-AF-01",
            "product_serial": "AF-2023-8812",
            "days_since_purchase": 100.0,
            "claim_amount_inr": -500.0,
            "photo_attached": "Y",
            "customer_prior_claims": 0,
            "claim_description": "unit not heating",
        })
        assert res.status_code == 422
