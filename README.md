# Kestrel Home — Warranty Fraud Decision-Support System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-27%2F27%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)]()

> **Production-minded warranty fraud decision-support application for Kestrel Home Appliances (Variant C).**  
> Identifies high-risk warranty claims prior to payout, optimizes investigation desk capacity (40 claims/month), and applies marginal expected-value triage against the ₹380 desk review fee.

---

## 📌 Executive Summary

Kestrel Home Appliances manufactures and distributes major household appliances across India via a network of ~380 authorized service centers, franchises, and freelance technicians.

Following a major operations policy shift on **1 May 2026** (fast-track auto-approval for claims under ₹2,000), fraudulent claims shifted overwhelmingly into micro-claims below ₹2,000 (97.2% of post-May fraud claims, average size ₹1,864).

### Key Business Realities & Architecture Choices:

1. **The 97% Accuracy Trap:** With fraud prevalence at only ~2.5%, a naive baseline predicting zero fraud trivially achieves **97.47% accuracy** while stopping ₹0 in fraud. The board's >97% accuracy KPI is therefore misleading. True operational value lies in **continuous risk ranking and net rupee recovery**.
2. **Investigation Capacity Constraints:** The claims desk can manually audit at most **40 claims per month**, incurring a customer goodwill friction cost of **₹380 per delayed genuine repair**.
3. **The Micro-Claim Economic Dilemma:** Auditing 40 claims/month blindly on validation data captured ₹13,131 across 7 frauds, but cost ₹30,400 in review fees (**net value: -₹17,269**; break-even precision requires ~20.4%).
4. **The Value-Weighted Solution:** We implemented **marginal expected-value screening** ($E[\text{Fraud Loss}] = \text{Risk Score} \times \text{Claim Amount}$). The system flags claims for review **only when expected recovery exceeds the ₹380 review cost**, eliminating negative-ROI reviews on low-dollar tickets.

---

## 🏗️ System Architecture

```
Incoming Warranty Claim (JSON / UI Form)
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│     1. Production Feature Pipeline (33 Features)       │
│  - Empirical Bayes Partner Smoothing (M=20)            │
│  - Serial Recycling Detection                          │
│  - Temporal Velocity & 7-Day / 30-Day Burst Metrics    │
│  - Claim-to-List-Price & Warranty Proximity Ratios     │
│  - Sanitized Fault Descriptions (Prompt Injection Safe)│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│     2. Calibrated Random Forest Model Pipeline         │
│  - Trained on Pre-May 2026 Claims (9,724 records)      │
│  - Zero Lookahead / Operational Leakage                │
│  - Excluded drifting partner-age features              │
│  - Output: Continuous Fraud Probability P(Fraud | X)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│     3. Economic Decision & Explanation Engine          │
│  - Expected Fraud Loss = P(Fraud | X) * Claim Amount   │
│  - Review Recommendation = (Expected Loss > ₹380)      │
│  - Risk Tiers: HIGH (>=0.40), MEDIUM (>=0.20), LOWER   │
│  - Top 3-5 Evidence-Based Reasons (Zero False Accusal) │
└─────────────┬────────────────────────────┬─────────────┘
              │                            │
              ▼                            ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│  4. FastAPI Microservice  │ │  5. Streamlit Review UI   │
│  - POST /predict          │ │  - Interactive Triage     │
│  - GET /health            │ │  - One-Click Presets      │
│  - GET /examples          │ │  - Resilient Fallback     │
└───────────────────────────┘ └───────────────────────────┘
```

---

## 🚀 Quickstart & Clean-Machine Setup

### Prerequisites

- Python `3.10`, `3.11`, `3.12`, or `3.13`
- Git

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Gnanendhra28/Kestrel-Home-Warranty-Claim-Review.git
cd Kestrel-Home-Warranty-Claim-Review

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install all runtime, API, UI, and testing dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Automated Test Suite (27/27 Tests)

```bash
pytest tests/ -v
```

_Executes integrity checks, feature pipeline tests, model constraints, expected value math, and API endpoint contracts._

### 3. Start the FastAPI Prediction Backend

```bash
# Runs on http://127.0.0.1:8000 by default (set HOST=0.0.0.0 for containers)
uvicorn src.api.main:app --host ${HOST:-127.0.0.1} --port ${PORT:-8000} --reload
```

- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 4. Start the Streamlit Employee Review Screen

```bash
streamlit run app/app.py --server.port 8501
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser to inspect incoming claims, test High/Medium/Low presets, view evidence reasons, and review desk ROI.

---

## 💻 API Usage Example

### Single-Claim Scoring (`POST /predict`)

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

### JSON Response

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

## 📊 Model Evaluation & Business Results

All models were evaluated using an **out-of-time temporal split** reflecting the real operational deployment window:

- **Training Period:** 2025-04-01 through 2026-04-30 (9,724 canonical claims)
- **Validation Period:** 2026-05-01 through 2026-06-30 (1,422 post-policy claims)
- **Test Period:** 2026-07-01 through 2026-09-30 (2,252 unlabelled claims scored in `outputs/predictions.csv`)

### Validation Performance:

| Metric                      | Selected Random Forest | Naive Baseline | Operational Interpretation                 |
| :-------------------------- | :--------------------: | :------------: | :----------------------------------------- |
| **ROC-AUC**                 |       **0.7059**       |     0.5000     | Solid discrimination across temporal shift |
| **PR-AUC**                  |       **0.0722**       |     0.0253     | ~3x baseline precision-recall envelope     |
| **Precision @ 40 (May)**    |       **7.50%**        |     2.50%      | 3 of 40 claims caught (vs 1 baseline)      |
| **Precision @ 40 (June)**   |       **10.00%**       |     2.50%      | 4 of 40 claims caught (vs 1 baseline)      |
| **Combined Precision @ 40** |       **8.75%**        |     2.50%      | 7 frauds caught out of 80 reviews          |
| **Combined Recall @ 40**    |       **19.44%**       |     5.56%      | Captures ~1 in 5 total frauds              |
| **Gross Fraud ₹ Captured**  |      **₹13,131**       |     ₹3,245     | **+₹9,886** incremental fraud stopped      |
| **Review Cost (80 audits)** |      **₹30,400**       |    ₹30,400     | ₹380 friction cost per investigated claim  |
| **Modeled Net Value**       |      **-₹17,269**      |    -₹27,155    | +₹9,886 improvement over random auditing   |
| **Inference Cost / Month**  |         **₹0**         |       ₹0       | Zero paid cloud APIs or commercial LLMs    |

---

## 📁 Repository Structure

```
.
├── app/
│   └── app.py                      # Streamlit employee triage dashboard
├── data/
│   ├── processed/                  # Processed data cache (.gitkeep)
│   └── raw/                        # Assignment reference files (train/test excluded per §10)
├── outputs/
│   ├── evaluation/                 # Error analysis & validation comparison CSVs
│   ├── features/                   # Feature manifest & engineering metadata
│   ├── models/                     # Saved pipeline (model_pipeline.joblib, feature_list.json)
│   ├── predictions.csv             # Official 2,252-row test predictions file
│   └── reports/                    # Complete business & technical reports
│       ├── business_context.md     # Phase 1 business mandate & operational constraints
│       ├── business_value_analysis.md # Phase 2B financial & sensitivity analysis
│       ├── data_audit_report.md    # Phase 1 data integrity & anomaly findings
│       ├── error_analysis.md       # Phase 2B false positive / false negative breakdown
│       ├── feature_engineering_report.md # Phase 2A feature dictionary & pipeline design
│       ├── feature_leakage_audit.md # Phase 2A audit on excluded leakage fields
│       ├── model_evaluation_report.md # Phase 2B model benchmark & selection report
│       ├── phase3_product_report.md # Phase 3 productization & service specification
│       ├── predictions_validation.md # Predictions distribution & sanity report
│       └── ritu_deshpande_memo.md  # 1-page executive memorandum for Head of D2C Ops
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI prediction & health check service
│   ├── business/
│   │   └── risk_engine.py          # Expected value screening & explanation engine
│   ├── data/
│   │   └── run_audit.py            # Data integrity & hash verification script
│   ├── features/
│   │   └── build_features.py       # Production feature extraction & bisection indices
│   └── models/
│       ├── generate_predictions.py # Official predictions generation script
│       └── train_evaluate.py       # Model training & temporal evaluation script
├── tests/
│   ├── test_data_integrity.py      # Schema & SHA-256 integrity tests (6 tests)
│   ├── test_features.py            # Leakage & deduplication tests (9 tests)
│   ├── test_models.py              # Model constraints & financial math tests (6 tests)
│   └── test_phase3.py              # Prediction integrity & API tests (6 tests)
├── FINAL_RUNBOOK.md                # Comprehensive operational runbook
├── requirements.txt                # Full runtime & testing dependency specifications
├── submission-form.md              # Completed 14-question assignment submission form
└── README.md                       # This documentation
```

---

## 🔒 Data Governance & Security Notice

In strict adherence to **Kestrel Warranty Operations Policy §10**:

- All raw customer, partner, and claims data (`train.csv`, `test_unlabelled.csv`, `partners.csv`, `ops-policy.pdf`, `email-thread.txt`) are classified as confidential and are **excluded from public Git tracking via `.gitignore`**.
- Adversarial prompt injection attacks embedded in historical fault descriptions were neutralized via deterministic canonical fault classification in `src/features/build_features.py`.
- Model inference is **100% self-hosted and local**, requiring zero external API keys or cloud credentials.

---

## 📄 License & Attribution

Designed and built for the **Kestrel Home — Warranty Claim Review Assignment (Variant C)**.  
Lead AI/ML Engineer: **Matta Gnanendhra** ([GitHub](https://github.com/Gnanendhra28))
