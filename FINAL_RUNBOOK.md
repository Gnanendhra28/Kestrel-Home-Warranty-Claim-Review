# Kestrel Home — Warranty Fraud Decision Support: Final Runbook

This runbook provides complete operational instructions for running tests, generating predictions, launching the FastAPI prediction backend, and running the Streamlit claims review dashboard.

---

## 1. Project Directory Structure

```
kestrel-warranty-fraud/
├── app/
│   └── app.py                      # Streamlit employee-facing review dashboard
├── data/
│   └── raw/                        # Original immutable source CSV, TXT, and PDF files
├── outputs/
│   ├── features/                   # Train and test feature matrices & manifest
│   ├── models/                     # Saved model pipeline, metadata & feature list
│   ├── predictions.csv             # Official 2,252-row test predictions file
│   └── reports/                    # Complete audit, leakage, model & business reports
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI single-claim scoring service
│   ├── business/
│   │   └── risk_engine.py          # Expected value screening & explanation engine
│   ├── data/
│   │   └── run_audit.py            # Phase 1 data audit & SHA-256 integrity verification
│   ├── features/
│   │   └── build_features.py       # Phase 2A leakage-safe feature engineering pipeline
│   └── models/
│       ├── train_evaluate.py       # Phase 2B model training & temporal evaluation
│       └── generate_predictions.py # Phase 3 prediction generation script
├── tests/
│   ├── test_data_integrity.py      # Raw data schema & SHA-256 hash checks (6 tests)
│   ├── test_features.py            # Feature engineering & leakage checks (9 tests)
│   ├── test_models.py              # Model performance, constraints & math checks (6 tests)
│   └── test_phase3.py              # Prediction integrity & API endpoints (6 tests)
├── FINAL_RUNBOOK.md                # This operational guide
├── README.md                       # Project architecture overview
└── submission-form.md              # Completed assignment submission questionnaire
```

---

## 2. Environment Setup & Installation

To run this project on a clean machine (requires Python >= 3.10, tested on Python 3.13):

### Clean-Machine Setup:
```bash
# 1. Navigate to the project root directory
cd path/to/kestrel-warranty-fraud

# 2. Create and activate a fresh virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install all required runtime dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Automated Test Execution

Run the complete 27-test validation suite covering all phases:
```bash
pytest tests/ -v
```
**Expected Outcome:** `27 passed in ~1.8s`.

---

## 4. Re-Generating Predictions

To re-score `outputs/features/test_features.csv` using the verified model:
```bash
python src/models/generate_predictions.py
```
**Output Location:** `outputs/predictions.csv` (2,252 rows, exactly matching `data/raw/sample_submission.csv`).

---

## 5. Starting the FastAPI Service

Launch the single-claim scoring backend (default: `127.0.0.1:8000`; set `HOST=0.0.0.0` for container/cloud deployment):
```bash
# Using uvicorn CLI (with configurable HOST and PORT):
uvicorn src.api.main:app --host ${HOST:-127.0.0.1} --port ${PORT:-8000}

# Or running the module directly:
python -m src.api.main
```
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Archetypal examples: `http://127.0.0.1:8000/examples`

---

## 6. Starting the Streamlit Review Dashboard

Launch the employee-facing triage screen on port 8501:
```bash
streamlit run app/app.py --server.port 8501
```
Open your browser to: `http://localhost:8501`

*(Note: The Streamlit app includes an automatic in-process fallback to score claims locally even if the FastAPI HTTP server is offline).*

---

## 7. Example API Request & Response

### Example Request (`POST /predict`):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "claim_id": "WC-DEMO-01",
       "submitted_at": "2026-07-20T14:30:00",
       "partner_id": "SP3033",
       "sku": "KH-AF-01",
       "product_serial": "AF-2023-8812",
       "days_since_purchase": 350,
       "claim_amount_inr": 4800.0,
       "photo_attached": "N",
       "customer_prior_claims": 3,
       "claim_description": "unit not heating tripping mcb"
     }'
```

### Expected JSON Response:
```json
{
  "claim_id": "WC-DEMO-01",
  "fraud_risk_score": 0.363695,
  "risk_level": "MEDIUM",
  "claim_amount_inr": 4800.0,
  "estimated_expected_fraud_value_inr": 1745.74,
  "review_cost_inr": 380.0,
  "review_recommended": true,
  "reasons": [
    "Claim amount (₹4,800) is 92.3% of product list price (₹5,199), well above normal repair component ratios.",
    "Claim submitted without supporting photographic verification, increasing documentation risk.",
    "Customer profile indicates high claim frequency (3 prior warranty claims).",
    "High monetary value (₹4,800) creates substantial financial loss exposure if erroneous.",
    "Claim submitted in the final phase of warranty coverage (350 days post-purchase on a 12-month policy)."
  ],
  "disclaimer": "DISCLAIMER: This assessment provides statistical risk-prioritization and decision-support guidance for internal triage only. A high score or review recommendation reflects elevated risk indicators and DOES NOT constitute confirmed fraud. Formal claim rejection or payout denial requires thorough physical investigation per Kestrel Warranty Operations Policy."
}
```

---

## 8. Where Predictions Are Located

The primary submission artifact is located at:
`outputs/predictions.csv`

Validation report:
`outputs/reports/predictions_validation.md`

---

## 9. Known Operational Limitations

1. **Low Baseline Fraud Prevalence:** Confirmed fraud accounts for only ~2.5% of total claims intake, imposing a high bar for manual review profitability.
2. **Post-May Sub-₹2,000 Shift:** Auto-approving claims below ₹2,000 concentrated 97% of fraud in micro-claims where a ₹380 review cost makes indiscriminate review unprofitable.
3. **Legacy Zoho Status Uncertainty:** Pre-migration CRM claims stored undecided investigations as `0` instead of blanks, introducing potential noise into historical labels.
4. **Partner Age Drift:** No partners were onboarded after mid-2025; partner age features were excluded to avoid distribution drift errors.

---

## 10. Submission Checklist

- [x] `outputs/predictions.csv` exists and contains 2,252 rows.
- [x] Predictions format and ordering match `sample_submission.csv` identically.
- [x] FastAPI service (`src/api/main.py`) running and tested.
- [x] Streamlit dashboard (`app/app.py`) running and tested.
- [x] 27 unit tests passing across all project phases.
- [x] `outputs/reports/ritu_deshpande_memo.md` created.
- [x] `submission-form.md` completed.
- [x] Zero external paid APIs or LLMs used.
