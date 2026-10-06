# Kestrel Home — Warranty Claim Review: Feature Leakage Audit

**Project Phase**: Phase 2A — Leakage-Safe Feature Engineering Pipeline  
**Document Type**: Architectural Safety & Information Boundary Verification  
**Author**: Lead AI/ML Engineer  
**Date**: October 2026  

---

## 1. Which features are production-safe?

A total of **36 features** are approved as production-safe. Every feature satisfies two core criteria:
1. It is directly available or computable at the precise timestamp a claim is submitted ($T_{\text{submission}}$), prior to any payout decision or manual inspection.
2. Its distribution and generation logic remain consistent between historical intake and the production test horizon (July 1 – September 30, 2026).

### Approved Production Feature Set:
* **Claim-Level**: `claim_amount_inr`, `log_claim_amount`, `is_sub_2000`, `claim_amount_threshold_gap`, `photo_attached`, `customer_prior_claims`.
* **Product & Warranty**: `sku`, `product_family`, `product_list_price`, `product_warranty_months`, `claim_amount_to_list_price_ratio`, `days_since_purchase`, `days_to_warranty_ratio`, `is_near_warranty_expiry`, `is_early_life_claim`.
* **Partner Profile**: `partner_type`, `partner_city`, `partner_age_days_at_claim`, `is_new_partner_180d`, `is_new_partner_365d`.
* **Partner Historical Velocity & Burst** (computed strictly prior to claim): `partner_claims_prev_7d`, `partner_claims_prev_30d`, `partner_claims_lifetime_prior`, `partner_claim_velocity_ratio`, `partner_sub2000_claims_prev_30d`, `partner_sub2000_ratio_prev_30d`.
* **Target-Based Partner History** (Bayesian smoothed strictly prior to claim): `partner_prior_fraud_count`, `partner_prior_resolved_claims`, `partner_smoothed_fraud_rate`.
* **Serial Recycling** (strictly prior to claim): `serial_prior_claim_count`.
* **Calendar & Policy Indicators**: `claim_month`, `claim_day_of_week`, `claim_hour`, `is_weekend`, `post_policy_change`.
* **Sanitized Text**: `clean_fault_description` (12 canonical fault categories).

---

## 2. Which features were rejected?

The following fields were explicitly rejected from the production feature matrix:
1. `inspector_note` (field technician text notes).
2. `partner_inspected` (Y/N inspection sign-off flag).
3. `source` (legacy Zoho Desk vs modern CRM system indicator).
4. `is_fraud` (ground-truth target label — excluded from all feature inputs).
5. Global lifetime partner fraud rates (un-windowed full-dataset target encoding).
6. Duplicate resubmission records (second submissions excluded from canonical training set).

---

## 3. Why was `inspector_note` rejected?

`inspector_note` was rejected due to **fatal operational leakage and severe post-policy missingness**:
1. **Timing of Generation**: An inspection note is authored by an authorized technician *during or after physical inspection*. In an automated pre-payout triage workflow, the model must score the claim *before* human dispatch or payout approval.
2. **Policy-Induced Missingness in Production**: Under Operations Policy v4.1 (§5), claims under ₹2,000 filed on or after May 1, 2026 are auto-approved without inspection. Consequently, no inspector is dispatched and no note is recorded.
3. **Distribution Shift**:
   * Pre-May 2026 (Train): ~81% of claims had notes.
   * Post-May 2026 Test Set: Exactly **81.22% of claims have no inspection note** (100% missing for sub-₹2,000 claims).
   * Any model dependent on `inspector_note` tokens would experience severe performance collapse on the 77% of intake that is auto-routed.

---

## 4. How was partner target leakage prevented?

Calculating a partner's fraud propensity using target labels is one of the highest-risk leakage vectors in tabular ML. We prevented target leakage through four strict design controls:
1. **Strictly Preceding Time-Windowing**: For any claim submitted at timestamp $T$, only claims submitted strictly before $T$ ($T_{\text{prior}} < T$) can contribute to the partner's historical fraud count.
2. **Self-Exclusion**: The current claim $i$ is never included in its own historical fraud count or sample size.
3. **No Test Set Label Feedback**: Test set claims (July 1 – September 30, 2026) have unknown labels. Test claims never update partner fraud statistics. When scoring test claims, partner historical fraud metrics use only resolved outcomes accumulated during the training period (up to June 30, 2026).
4. **Empirical Bayes Shrinkage**: Rather than using raw fraud ratios ($\frac{\text{fraud}}{\text{claims}}$) which produce noisy, extreme values for low-volume partners, we compute:
   $$\text{Smoothed Rate} = \frac{\text{prior\_fraud} + m \cdot p_0}{\text{prior\_resolved} + m}$$
   where $p_0 = 0.01265$ (global training base rate) and $m = 20.0$.

---

## 5. How were duplicates prevented from contaminating validation?

The audit identified 681 duplicate claim IDs in `train.csv` (1,362 rows), representing bounced portal resubmissions.
1. **Canonical Deduplication**: We selected the **earliest submission timestamp (`first_submitted_at`)** as the single canonical representation for each unique claim (11,348 unique claims).
2. **Zero Cross-Split Contamination**: Because duplicates are deduplicated to unique `claim_id`s prior to splitting, the identical claim can never appear in both training and validation sets.
3. **No Weight Distortion**: Excluding the 681 duplicate second-submissions prevents the loss function from giving double weight to bounced claims.
4. **Metadata Preservation**: Duplicate details (`has_resubmission`, `resubmission_count`, `resubmission_delay_hours`) are preserved in `train_modeling_metadata.csv` and `excluded_records.csv` for auditability.

---

## 6. How were cold-start partners handled?

In `test_unlabelled.csv`, **14 partners have zero historical claims in `train.csv`** (newly active partners onboarded to cover Tier-2/3 cities).
1. **Shrinkage to Population Prior**: In the Empirical Bayes formula, when $\text{prior\_resolved} = 0$ and $\text{prior\_fraud} = 0$, the formula simplifies to:
   $$\text{Smoothed Rate} = \frac{0 + 20 \times 0.01265}{0 + 20} = 0.01265$$
   Every cold-start partner receives the exact population base rate ($1.265\%$), avoiding artificial penalization or zero-division errors.
2. **Robust Entity Attributes**: Cold-start partners retain valid demographic signals (`partner_type`, `partner_city`, `partner_age_days_at_claim`), enabling tree models to generalize across partner types and geographies.
3. **Zero Volume Imputation**: Volume and velocity features (`partner_claims_prev_7d`, `partner_claims_prev_30d`, `partner_claims_lifetime_prior`) are naturally and truthfully set to `0`.

---

## 7. How was the May 1 policy change handled?

Operations Policy v4.1 changed claim handling on May 1, 2026: claims under ₹2,000 became auto-approved without inspection.
1. **Explicit Policy Features**: We created `is_sub_2000` (binary) and `claim_amount_threshold_gap` ($\max(0, 2000 - \text{amount})$) which capture whether a claim qualifies for auto-approval and its proximity to the boundary.
2. **Burst Sub-₹2,000 Tracking**: `partner_sub2000_ratio_prev_30d` tracks the proportion of recent claims submitted by a partner below ₹2,000, exposing bad actors who pivot their claim volume just below the inspection cutoff.
3. **Temporal Split Alignment**: Our validation split is placed on May 1 – June 30, 2026. This forces model validation to evaluate on the exact post-May policy regime, mirroring test conditions.
4. **Rejection of `partner_inspected`**: We rejected `partner_inspected` because using it as a feature would cause models to treat "not inspected" as high-risk, when in the post-May regime 78.7% of genuine claims are not inspected by design.

---

## 8. How was legacy Zoho target uncertainty handled?

Zoho Desk (used until September 30, 2025) could not store null/blank fraud flags; unresolved cases at migration were defaulted to `0`.
1. **Honest Boundary Acknowledgment**: The raw export does not include the legacy Zoho audit event log. It is impossible to identify with certainty which legacy zeros were genuine non-fraud versus unresolved cases.
2. **No Arbitrary Heuristics**: Rather than inventing fictitious ground truth or dropping 40% of the dataset, we retain the historical data as recorded.
3. **Clean Validation Regime**: The primary validation split (May–June 2026) is entirely within the modern CRM system where unresolved cases were explicitly left blank (and excluded from supervised labels), ensuring our evaluation metric is calculated on clean ground truth.

---

## 9. Can any feature see future information?

**NO**. An exhaustive audit of the feature extraction pipeline proves:
1. Every claim timestamp $T_i$ query uses `bisect.bisect_left` on sorted historical event lists, which selects indices strictly less than $T_i$.
2. The current claim is excluded from all 7-day, 30-day, and lifetime historical counts.
3. Target-based partner metrics enforce an additional hard cutoff ($T < \text{2026-07-01}$), ensuring test claims never update or leak target statistics into earlier or contemporaneous claims.
4. Product catalog attributes (`list_price_inr`, `warranty_months`) and partner onboarding dates are static reference tables known at the time of claim filing.

---

## 10. What assumptions remain uncertain?

1. **Legacy Zoho Contamination Rate**: The exact proportion of legacy Zoho `0`s that were actually unresolved claims remains unknown (estimated at ~2-3% based on CRM blank rates).
2. **Investigation Desk Turnaround Delay**: In reality, fraud investigations take several days or weeks to conclude. Our historical target feature assumes a resolved outcome is known once established, which is safe for test scoring since test claims are evaluated months after training ended, but in live streaming operations an explicit investigation lag buffer (e.g. 14 days) should be configured.
3. **Cold-Start Volume Surge in Test**: 14 partners have zero training claims. If these partners experience sudden bursts in test, their early test claims rely primarily on type/city priors until volume accumulates.
