# Kestrel Home — Warranty Claim Review: Business Context & Domain Specification

**Project Variant**: Variant C (Lightweight Warranty Fraud Decision-Support Application)  
**Document Type**: Phase 1 Business Context & Requirements Synthesis  
**Date**: October 2026  
**Status**: Read-Only Discovery Complete  

---

## 1. Executive Summary & Business Objective

Kestrel Home Appliances manufactures and distributes consumer home appliances across Tier-1, Tier-2, and Tier-3 Indian cities (Air Fryers, Mixer Grinders, Water Purifiers, Robot Vacuums, Induction Cooktops, Ceiling Fans, and Room Heaters). Warranty and extended-warranty ("Kestrel Shield") repairs are fulfilled through a network of ~380 service partners (authorised service centres, franchises, and freelance technicians).

Partners submit warranty claims for labor and replacement parts. Over the past year, Kestrel expanded rapidly into smaller cities by onboarding approximately 60 new service partners. Concurrently, warranty claim volumes have risen, and suspected fraudulent claim submissions have created financial leakage.

### Core Objectives:
1. **Decision Support**: Build an AI/ML decision-support system to flag high-risk warranty claims *prior to payout*.
2. **Investigation Desk Prioritization**: Surface the highest-yield claims for manual audit within strict capacity constraints.
3. **Explainability**: Provide human-interpretable rationales (risk drivers, claim context) for why a claim received its risk score so investigators can verify claims efficiently.
4. **Economics-First Evaluation**: Evaluate models not merely on academic statistics, but on net financial savings (rupees of fraud prevented per claim investigated, net of review goodwill costs).

---

## 2. Stakeholder Perspectives & Requirements

The cross-functional email thread reveals critical divergences in perspective that the ML solution must reconcile:

| Stakeholder | Role | Stated Position / Request | ML & Business Analysis |
| :--- | :--- | :--- | :--- |
| **Ritu Deshpande** | Head of D2C Operations | Requested Board KPI: **Accuracy > 97%**.<br>Suspects **newer service partners** are responsible ("let's have the data say it"). | **The 97% Accuracy Trap**: With an observed fraud rate of only 1.21% (145 fraud cases out of 12,029 claims), a naive baseline predicting `is_fraud = 0` for all claims achieves **97.01% accuracy** while stopping zero fraud! Overall accuracy is deeply misleading; precision, recall, PR-AUC, and financial savings at the operating threshold are the true operational metrics.<br>**Partner Hypothesis**: Empirical testing confirms newer partners (onboarded $\ge$ 2025-06-30) have a higher fraud rate (6.99% vs 0.97%), but the highest absolute volumes of fraud come from a small subset of persistent outlets regardless of age (e.g. franchises onboarded in 2021-2023). |
| **Farhan Sheikh** | Finance Controller | Primary KPI: **Fraud prevented per claim checked in rupees**.<br>Emphasizes that every false positive delays genuine customer repairs and incurs goodwill costs. | Confirms that evaluation must be framed around **Expected Value / Profit Optimization** rather than arbitrary probability cutoffs. Model evaluation must compute net rupees saved at the 40 claims/month review capacity. |
| **Meenal Joshi** | Service Desk Manager | Cautions that **most new partners in smaller cities are legitimate** and critical for coverage.<br>Notes: *"Since the May change the small claims have exploded - my team sees the same few outlets again and again."* | Validates the need to avoid blanket penalization of new partners. Identifies a critical behavioral shift following the 1 May 2026 policy change (claims under ₹2,000 auto-approved without inspection), where bad actors exploit the ₹2,000 threshold. |
| **Tanmay Kulkarni** | Data & IT Admin | Provided export details: partners resubmit bounced claims; serials are manually typed (messy); system migration subtleties. | Explains key data semantics: blank `is_fraud` in CRM vs zero `is_fraud` in legacy Zoho, time gaps in duplicate claims, and IST timestamp formatting. |

---

## 3. Operational Constraints & Workflow Rules

### 3.1 Investigation Desk Capacity
* **Monthly Audit Capacity**: The claims investigation desk can audit **at most 40 claims per month**.
* **Test Set Reality**: Test data contains 2,252 claims over 3 months (July – September 2026), representing approximately 750 claims/month.
* **Operating Point**: The team can review only ~40 / 750 $\approx$ **5.3% of claims** each month (or 120 claims across the entire 3-month test period).
* **Implication**: Any model thresholding must support rank-ordering (ranking by expected fraud payoff or risk score) to select the top 40 claims per month.

### 3.2 Policy Shift: The 1 May 2026 Rule
According to Kestrel Operations Policy v4.1 (§5):
* **Before 1 May 2026**: Every claim required partner inspection sign-off (`partner_inspected = 'Y'`).
* **From 1 May 2026**: Claims **under ₹2,000 are auto-approved without inspection** to reduce customer turnaround time. Claims $\ge$ ₹2,000 still require inspection.
* **Data Impact**:
  * In training data (April 2025 – June 2026), 83.7% of claims were inspected.
  * In test data (July 2026 – September 2026), 77.0% of claims are under ₹2,000. Exactly **0% of claims under ₹2,000 in test have partner inspection**, causing `partner_inspected = 'N'` for 78.7% of test claims and `inspector_note` to be blank in 81.2% of test claims.
  * Gaming Risk: Fraudulent partners rapidly adapted to claim just under ₹1,999 to bypass human inspection.

---

## 4. Financial Parameters & Decision Economics

Decision economics are defined in Operations Policy v4.1 (§4) and the management thread:

| Parameter | Value | Operational Context |
| :--- | :--- | :--- |
| **False Positive Cost ($C_{FP}$)** | **₹380** | Average customer goodwill voucher issued when a genuine claim is held for investigation, causing repair delays. |
| **False Negative Cost ($C_{FN}$)** | **Claim Amount (`claim_amount_inr`)** | A fraudulent claim that is auto-approved or missed costs Kestrel the full payout amount (mean ₹2,620 in train, max ₹26,728). |
| **Blended Service Contact Cost** | **₹260** | Operational handling cost per claim contact. |
| **True Positive Benefit ($B_{TP}$)** | **Claim Amount (`claim_amount_inr`)** | Payout avoided by flagging and successfully confirming fraud. |
| **Net Financial Savings** | $\sum_{TP} \text{Amount} - (40 \times \text{months} \times ₹380)$ | Total fraud rupees stopped minus total investigation friction/goodwill costs incurred on reviewed claims. |
| **Finance Metric** | $\frac{\text{Net Fraud Rupees Stopped}}{\text{Claims Investigated}}$ | Rupees saved per reviewed claim (must be significantly positive). |

---

## 5. Systems, Timestamps, and Target Semantics

### 5.1 System Migration (Zoho Desk $\to$ Kestrel CRM)
* **Legacy System (`legacy_zoho`)**: Used until 30 September 2025. Records migrated in bulk.
* **New System (`crm`)**: Used from 1 October 2025 onwards.
* **Source Column**: Explicitly identifies system origin (`legacy_zoho` vs `crm`).

### 5.2 Target Definition & Migration Nuance (`is_fraud`)
* **Values**:
  * `1`: Confirmed fraud.
  * `0`: Genuine / not fraud.
  * `""` (blank): Undecided case still under investigation at time of export.
* **CRITICAL ZOHO QUIRK**:
  * Zoho Desk database schemas could not store null/blank values for the fraud status flag.
  * During migration, **undecided legacy claims were converted to 0**.
  * In CRM (post-Oct 2025), undecided cases are preserved as blanks (215 rows in train).
  * **Implication**: Rows with `is_fraud == 0` from `legacy_zoho` have label noise (some unresolved claims masquerading as 0). CRM records with blank target must be handled deliberately (not blindly treated as 0).

### 5.3 Timestamps
* All export timestamps are in **Indian Standard Time (IST)** as displayed in the CRM.
* Exception: Legacy Zoho resolution events were originally logged in UTC, but the export features (`submitted_at`) are reported in IST.

---

## 6. Data Governance & Security Restrictions

* **Operations Policy §10**: Kestrel customer and operational data is confidential.
* **Prohibition**: Data must **NOT** be committed to public repositories, uploaded to public cloud buckets, or exposed externally.
* **Local Sandbox**: All audit, modeling, and application code must reside in local, private workspace directories.

---

## 7. Submission Specifications

* **Output File**: `outputs/predictions/predictions.csv` (matching `data/raw/sample_submission.csv` format).
* **Schema**: Exactly two columns: `claim_id,score`.
* **Row Count**: Exactly 2,252 rows matching `test_unlabelled.csv`.
* **Order**: Must match the exact row sequence of `sample_submission.csv`.
* **Score Semantics**: Calibrated fraud probability or risk score in the interval $[0.0, 1.0]$.
