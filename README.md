# Kestrel Home — Warranty Claim Review Decision-Support System

**Variant**: Variant C  
**Project Phase**: Phase 2A — Leakage-Safe Feature Engineering Pipeline  
**Status**: Feature engineering pipeline implemented, validated, and executed. All raw source datasets remain untouched. No ML model training, prediction generation, or API/UI building has occurred yet.  

---

## 1. Project Purpose

This project develops a lightweight, production-minded warranty fraud decision-support application for Kestrel Home Appliances. Kestrel manufactures and distributes major home appliances (Air Fryers, Mixer Grinders, Water Purifiers, Robot Vacuums, Induction Cooktops, Ceiling Fans, and Room Heaters) serviced across India via a partner network of ~380 service centres, franchises, and freelance technicians.

The ultimate objective of the application is to:
1. Identify high-risk warranty claims *prior to payout*.
2. Prioritize claims for the claims investigation desk, which has a strict operational review capacity of **at most 40 claims per month**.
3. Maximize financial return by optimizing **fraud rupees prevented per claim checked** (accounting for customer goodwill costs on delayed genuine repairs).
4. Provide explainable risk drivers for human investigators.
5. Provide a single-record scoring API and review dashboard.

---

## 2. Source Files Inventory

Original source files are located in the engagement pack and mirrored immutably in `data/raw/`:

* `train.csv`: Historical warranty claims (April 2025 – June 2026, 12,029 records) with investigation outcomes.
* `test_unlabelled.csv`: Recent warranty claims (July 2026 – September 2026, 2,252 records) requiring risk scoring.
* `partners.csv`: Service partner registry (380 partners) with city, onboarding date, and partner type.
* `products.csv`: Product catalog (21 SKUs across 7 appliance families) with list price and warranty duration.
* `sample_submission.csv`: Benchmark submission schema (2,252 records, columns: `claim_id,score`).
* `ops-policy.pdf`: Kestrel Operations Policy v4.1 detailing approval workflows, costs, and audit capacity.
* `email-thread.txt`: Cross-functional stakeholder messages outlining requirements, operational bottlenecks, and domain nuances.
* `README.txt`: Original data pack field descriptions and source notes.

> **Data Governance Note**: According to Kestrel Operations Policy §10, all customer, partner, and claims data is strictly confidential and must never be uploaded or published to public repositories.

---

## 3. Phase 2A Feature Engineering Architecture

Phase 2A built and executed an end-to-end, strictly leakage-safe feature engineering pipeline (`src/features/build_features.py`) producing 36 approved production features.

### 3.1 Production-Safe Feature Philosophy
Every production feature represents information that is **guaranteed to be known at claim submission time** ($T_{\text{submission}}$), prior to any payout decision or manual inspection.
* **Excluded Operational Leakage**: Fields like `inspector_note` and `partner_inspected` were strictly excluded because they represent post-inspection activities. Furthermore, post-May 1, 2026 claims under ₹2,000 are auto-approved without inspection, leaving `inspector_note` missing in 81.2% of test claims.
* **Sanitized Text**: `claim_description` is normalized to 12 canonical fault categories, neutralizing 5 adversarial prompt injections discovered in raw training records.

### 3.2 Temporal Leakage Prevention & Windowing
* All partner burst metrics (`partner_claims_prev_7d`, `partner_claims_prev_30d`, `partner_claims_lifetime_prior`) and serial recycling counts are indexed chronologically. Lookups use strict bisection ($T_{\text{prior}} < T_{\text{current}}$).
* The current claim is excluded from its own historical metrics.
* **Target-Based Partner Features**: Partner historical fraud statistics use strictly prior resolved outcomes ($T_{\text{prior}} < T_{\text{current}}$ and $T_{\text{prior}} < \text{2026-07-01}$) smoothed with Empirical Bayes shrinkage toward the population prior ($0.01265$). Test set claims never update fraud rates, preventing lookahead leakage.

### 3.3 Duplicate Claim Resolution
* Bounced resubmissions (681 duplicate claim pairs in `train.csv`) were resolved by adopting the **earliest submission timestamp (`first_submitted_at`)** as the canonical record (11,348 unique claims).
* Duplicate second submissions are isolated in `outputs/features/excluded_records.csv` to prevent loss function weight distortion and train/validation cross-contamination.

### 3.4 Target Label Handling
* Confirmed fraud (`is_fraud = 1`, 141 canonical claims) and confirmed genuine (`is_fraud = 0`, 11,005 canonical claims) form the supervised training set (11,146 rows).
* Undecided open investigations (202 unique canonical claims with blank target) are excluded from supervised training and logged in `outputs/features/excluded_records.csv`.
* Legacy Zoho Desk label uncertainty (open cases defaulting to 0 prior to October 2025) is documented as a known boundary condition.

### 3.5 Policy Shift Treatment (1 May 2026)
* Sub-₹2,000 auto-approval rule is explicitly modeled via `is_sub_2000`, `claim_amount_threshold_gap`, and `partner_sub2000_ratio_prev_30d` (tracking partner burst gaming around the threshold).
* The temporal validation split (May 1 – June 30, 2026) isolates the post-policy regime for unbiased evaluation.

---

## 4. How to Run Phase 2A Pipeline & Tests

All code relies purely on the Python standard library with zero external dependency requirements.

### Run the Feature Pipeline
From the project root:
```bash
python3 src/features/build_features.py
```

### Run Comprehensive Test Suite
```bash
python3 -m unittest discover -s tests -v
```

---

## 5. Next Steps (Phase 2B Roadmap)
1. **Model Selection & Training**: Train cost-sensitive gradient boosted decision trees (LightGBM/XGBoost) using temporal validation (Train: April 2025 – April 2026; Val: May – June 2026).
2. **Economic Optimization**: Calibrate thresholding to maximize net fraud rupees prevented at the 40 claims/month review capacity.
3. **Scoring API & Review Dashboard**: Build FastAPI endpoint and review UI with SHAP explanations.
