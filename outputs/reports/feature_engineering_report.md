# Kestrel Home — Warranty Claim Review: Feature Engineering Report

**Project Variant**: Variant C (Lightweight Warranty Fraud Decision-Support Application)  
**Project Phase**: Phase 2A — Leakage-Safe Feature Engineering Pipeline  
**Date**: October 2026  
**Status**: Pipeline Built, Tested, and Executed (No ML models trained yet)  

---

## 1. Executive Summary

Phase 2A implements a production-minded, strictly leakage-safe feature engineering pipeline for Kestrel Home Appliances' warranty claim review system. The system's primary operational purpose is to score warranty claims **before payout**, prioritizing the highest-risk claims for manual review by the claims desk (capacity: $\le$ 40 reviews/month).

To guarantee production validity on the out-of-time test horizon (July 1 – September 30, 2026), every single feature was engineered under the **Zero-Leakage Principle**:
1. All features must be genuinely available at the moment a partner files a warranty claim.
2. No post-investigation artifacts (e.g. `inspector_note`) are admitted into the production feature matrix.
3. Historical aggregates and velocity metrics use strictly earlier events ($T_{\text{prior}} < T_{\text{current}}$); the current claim is never counted toward its own historical metrics.
4. Target-based partner aggregates utilize strictly prior resolved outcomes with Empirical Bayes shrinkage toward the global baseline, eliminating lookahead and small-sample variance.
5. Duplicate resubmissions were forensically resolved by establishing a deterministic **canonical first-submission** representation.

---

## 2. Duplicate Claims & Resubmission Resolution

### 2.1 Forensic Discovery
The Phase 1 audit revealed that 681 unique `claim_id` values appear twice in `train.csv` (1,362 rows out of 12,029 total rows). Conversely, `test_unlabelled.csv` contains exactly 2,252 rows with zero duplicates.

Forensic analysis of the 681 duplicate pairs in train demonstrated:
* **Feature Invariance**: In 100% of cases (681 / 681), all claim attributes (`partner_id`, `sku`, `product_serial`, `claim_amount_inr`, `days_since_purchase`, `photo_attached`, `claim_description`, `customer_prior_claims`, and `is_fraud`) were completely identical.
* **Timestamp Delta**: Only `submitted_at` differed. The time gap between submission 1 and submission 2 ranged from **0.35 days (8.4 hours) to 5.00 days** (median: 3.00 days).
* **Operational Cause**: As noted by IT Admin Tanmay Kulkarni, service partners resubmit claims when an earlier submission bounces or encounters a portal timeout.

### 2.2 Canonical Selection Decision: First Submission vs Latest Submission
We evaluated whether to select the first submission or the latest submission as the canonical record:

* **Arguments Against "Latest Submission"**:
  1. **Temporal Lookahead**: A later resubmission timestamp moves the decision point forward by 1 to 5 days. Calculating partner velocity, burst counts, or 30-day volumes using the later timestamp introduces activity that occurred *after* the claim was actually initiated.
  2. **Policy Boundary Contamination**: Exactly 10 duplicate claim pairs in `train.csv` span the May 1, 2026 policy shift (first submitted in late April 2026, second submitted in early May 2026). Using the second timestamp would misclassify pre-policy claims as post-policy claims.
  3. **Production Alignment**: In a production pre-payout gate, the review queue triggers the instant a claim is first received. Evaluating the initial submission reflects the real-world operational decision point.

* **Decision**: We establish the **earliest submission timestamp (`first_submitted_at`)** as the deterministic canonical claim representation:
  * Canonical unique training claims: **11,348 records**.
  * The 681 second-submission records are excluded from the supervised training feature matrix and preserved in `outputs/features/excluded_records.csv` with reason `duplicate_resubmission_second_entry`.
  * Resubmission indicators (`has_resubmission`, `resubmission_count`, `resubmission_delay_hours`) are captured in `outputs/features/train_modeling_metadata.csv` for post-hoc operational diagnostics.

---

## 3. Target Label Handling & Legacy System Uncertainties

### 3.1 Supervised Target Selection
* **Ground Truth Definitions**:
  * `is_fraud = 1`: Confirmed fraudulent claims (141 unique canonical claims).
  * `is_fraud = 0`: Confirmed genuine / non-fraud claims (11,005 unique canonical claims).
* **Exclusion of Undecided Records**:
  * 202 unique canonical claims (215 raw rows) have blank `is_fraud` values. These represent active, unresolved investigations at the time of data export.
  * These 202 claims are excluded from the supervised training feature matrix (`train_features.csv`) and logged in `outputs/features/excluded_records.csv` with reason `open_case_undecided_target`.
  * Supervised modeling training set: **11,146 canonical labeled claims**.

### 3.2 Legacy Zoho Target Uncertainty
* **Operational Reality**: Operations Policy v4.1 (§9) and Tanmay's email disclose that until September 30, 2025, operations ran on legacy Zoho Desk. Zoho database schemas could not store null/blank values for fraud outcomes. Consequently, open/unresolved cases at migration were converted to `0`.
* **Empirical Observation**: In `train.csv`, `legacy_zoho` records (April – September 2025) have exactly 0 blank targets, whereas `crm` records have 215 blanks (~3.0% per month).
* **Resolution**: The raw data export does not contain the legacy Zoho resolution audit log. It is forensically impossible to reliably distinguish which legacy zeros were genuine non-fraud versus defaulted open cases without inventing arbitrary heuristics. Rather than fabricating labels, we:
  1. Document this label noise limitation explicitly.
  2. Structure our primary validation split on post-migration CRM data (May 1 – June 30, 2026) where target labels are verified clean.

---

## 4. Production Feature Architecture

A total of **36 production-safe features** were engineered across 7 functional groups. Every feature is cataloged in `outputs/features/feature_manifest.csv`.

### 4.1 Claim-Level Features (6 features)
1. `claim_amount_inr`: Declared repair reimbursement amount in Indian Rupees.
2. `log_claim_amount`: Natural log transformation $\ln(1 + \text{amount})$ to stabilize right-skewed repair costs.
3. `is_sub_2000`: Binary indicator ($1$ if $\text{amount} < ₹2,000$, else $0$).
4. `claim_amount_threshold_gap`: Proximity to the policy threshold: $\max(0, 2000.0 - \text{amount})$ if sub-2000, else $0.0$. Captures strategic clustering just beneath the auto-approval cutoff.
5. `photo_attached`: Binary indicator ($1$ if diagnostic photograph uploaded, else $0$).
6. `customer_prior_claims`: Historical CRM warranty claim count for the customer account.

### 4.2 Product & Warranty Engineering (9 features)
Joined from `products.csv` using `sku`:
7. `sku`: Categorical product identifier (21 unique SKUs).
8. `product_family`: Appliance category (7 families: Air Fryer, Ceiling Fan, Induction Cooktop, Mixer Grinder, Robot Vacuum, Room Heater, Water Purifier).
9. `product_list_price`: Catalog retail price in rupees (₹1,999 to ₹29,698).
10. `product_warranty_months`: Manufacturer warranty term (12 or 24 months).
11. `claim_amount_to_list_price_ratio`: Ratio of claimed amount to product list price ($\frac{\text{claim\_amount}}{\text{list\_price}}$). Identifies claims requesting near-replacement costs for minor parts.
12. `days_since_purchase`: Elapsed days from customer invoice to claim submission.
13. `days_to_warranty_ratio`: Warranty wear fraction ($\frac{\text{days\_since\_purchase}}{\text{warranty\_months} \times 30.5}$).
14. `is_near_warranty_expiry`: Flag indicating claim filed in the final 15% of the warranty term ($\text{ratio} \ge 0.85$).
15. `is_early_life_claim`: Flag indicating infant mortality claim filed within 30 days of purchase ($\le 30$ days).

### 4.3 Partner Demographic Profile (5 features)
Joined from `partners.csv` using `partner_id`:
16. `partner_type`: Entity structure (`authorised_service_centre`, `franchise`, `freelance_technician`).
17. `partner_city`: Partner operational hub (18 cities across India).
18. `partner_age_days_at_claim`: Operational tenure: $(\text{claim\_date} - \text{onboarded\_date})\text{ in days}$.
19. `is_new_partner_180d`: Flag for partners onboarded $\le 180$ days prior to claim.
20. `is_new_partner_365d`: Flag for partners onboarded $\le 365$ days prior to claim.

### 4.4 Partner Velocity & Burst Metrics (6 features)
Computed using a temporal event index strictly prior to current claim ($T_{\text{prior}} < T_{\text{current}}$):
21. `partner_claims_prev_7d`: Claims submitted by this partner in preceding 7 calendar days.
22. `partner_claims_prev_30d`: Claims submitted by this partner in preceding 30 calendar days.
23. `partner_claims_lifetime_prior`: Total cumulative historical intake for this partner prior to this claim.
24. `partner_claim_velocity_ratio`: Surge factor: $\frac{\text{claims\_prev\_7d}}{(\text{claims\_prev\_30d} / 4.0) + 1.0}$. Measures whether the weekly pace is abnormally elevated relative to recent monthly baseline.
25. `partner_sub2000_claims_prev_30d`: Volume of claims $< ₹2,000$ filed by partner in past 30 days.
26. `partner_sub2000_ratio_prev_30d`: Share of sub-₹2,000 claims: $\frac{\text{sub2000\_claims\_30d}}{\max(\text{total\_claims\_30d}, 1)}$.

### 4.5 Target-Based Partner History with Empirical Bayes Shrinkage (3 features)
Strictly time-aware target aggregation:
27. `partner_prior_fraud_count`: Confirmed fraud claims for this partner strictly preceding current claim ($T_{\text{prior}} < T_{\text{current}}$ and $T_{\text{prior}} < \text{2026-07-01}$).
28. `partner_prior_resolved_claims`: Historical resolved audit sample size for this partner prior to current claim.
29. `partner_smoothed_fraud_rate`: Empirical Bayes smoothed fraud rate:
   $$\text{Rate} = \frac{\text{prior\_fraud} + m \cdot p_0}{\text{prior\_resolved} + m}$$
   where $p_0 = 0.01265$ (global training base rate) and $m = 20.0$ (smoothing weight).
   * **Cold-Start Guarantee**: For any partner with 0 prior claims or unobserved in training (including the 14 new test partners), the smoothed fraud rate equals exactly $0.01265$ (the population baseline).

### 4.6 Serial Recycling Detection (1 feature)
30. `serial_prior_claim_count`: Number of times the specific `product_serial` was previously claimed strictly prior to $T_{\text{current}}$. Identifies syndicates recycling valid serial numbers across repeated tickets.

### 4.7 Calendar & Policy Indicators (5 features)
31. `claim_month`: Month of submission (1 to 12).
32. `claim_day_of_week`: Day of week (0=Monday, 6=Sunday).
33. `claim_hour`: Hour of submission in IST (0 to 23).
34. `is_weekend`: Binary flag for weekend submissions.
35. `post_policy_change`: Binary indicator whether claim was submitted on or after May 1, 2026.

### 4.8 Sanitized Text Representation (1 feature)
36. `clean_fault_description`: Standardized appliance failure category mapped to 12 canonical domain classes (`unit not heating`, `motor not running`, `tripping mcb`, `filter indicator stuck`, `remote not working`, `blade jammed`, `loud noise while running`, `power button not working`, `display not working`, `water leaking`, `burning smell`, `not charging`).
   * **Adversarial Sanitization**: Strips out all 5 prompt injection attacks discovered in training descriptions, preventing model contamination.

---

## 5. Excluded Features & Leakage Audit Summary

The following fields were strictly excluded from the production feature matrix:
* **`inspector_note` (REJECTED)**: Post-inspection field notes. In production, 77% of claims are sub-₹2,000 and auto-approved without inspection; in test data, `inspector_note` is missing in 81.2% of claims. Admitting it creates massive production failure.
* **`partner_inspected` (REJECTED)**: Suffers severe policy-induced distribution shift (83.7% 'Y' in train vs 21.3% 'Y' in test). Post-May claims under ₹2,000 are auto-approved without inspection.
* **`source` (REJECTED)**: Migration system flag (`legacy_zoho` vs `crm`). Test intake is 100% CRM. Using `source` causes models to memorize legacy artifacts.
* **`is_fraud` (TARGET ONLY)**: Retained only as the training target label. Excluded from all feature input sets.
* **Full-Dataset Lifetime Partner Aggregates (REJECTED)**: Calculating lifetime fraud rates across the entire training dataset leaks future claims into past records.

---

## 6. Generated Output Files

All feature datasets were successfully written to `outputs/features/`:

| File Name | Row Count | Column Count | Description |
| :--- | :---: | :---: | :--- |
| `train_features.csv` | 11,146 | 38 | Supervised training matrix (`claim_id`, `is_fraud`, + 36 features). |
| `test_features.csv` | 2,252 | 37 | Unlabelled test feature matrix (`claim_id` + 36 features). |
| `feature_manifest.csv` | 42 | 7 | Comprehensive metadata for all 36 production + 6 rejected features. |
| `train_modeling_metadata.csv` | 11,348 | 8 | Resubmission metrics, timestamps, and split assignments for all unique claims. |
| `excluded_records.csv` | 883 | 7 | Audit trail of 681 duplicate resubmissions + 202 undecided claims. |
| `partner_history_diagnostics.csv` | 380 | 9 | Complete profile of all 380 partners with historical volumes and cold-start flags. |

---

## 7. Temporal Validation Split Recommendation for Phase 2B

To evaluate future models without lookahead bias while mirroring test set dynamics:
* **Training Window**: `2025-04-01` to `2026-04-30` (9,724 canonical labeled claims; fraud rate 1.08%).
* **Validation Window**: `2026-05-01` to `2026-06-30` (1,422 canonical labeled claims; fraud rate 2.53%).
  * Captures the identical post-May 1, 2026 sub-₹2,000 auto-approval regime.
* **Final Test Window**: `2026-07-01` to `2026-09-30` (2,252 unlabelled claims).
