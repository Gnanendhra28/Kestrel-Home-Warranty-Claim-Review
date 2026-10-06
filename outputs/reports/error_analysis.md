# Kestrel Home — Warranty Claim Review: Forensic Error Analysis

**Project Phase**: Phase 2B — Forensic Validation Error Analysis  
**Selected Model**: Random Forest (depth=6, Balanced)  
**Dataset**: Prospective Validation Set (`2026-05-01` to `2026-06-30`, 1,422 claims, 36 confirmed frauds)  
**Artifact Reference**: [`outputs/evaluation/error_analysis.csv`](file:///Users/gnanendhrajoy/Desktop/Kestrel%20Home/kestrel-warranty-fraud/outputs/evaluation/error_analysis.csv)  

---

## 1. Executive Summary

Forensic error analysis was conducted on the 1,422 validation claims scored by the selected Random Forest model. We examined model predictions through two complementary operational lenses:
1. **The Operational Top-40 Review Queue**: The top 40 highest-risk claims per month (80 claims total across May and June 2026).
2. **The Conventional 0.5 Probability Cutoff**: Binary classification behavior under equal thresholding.

### Validation Confusion Matrix Breakdown:
* **Under Monthly Top-40 Queue (80 claims reviewed out of 1,422 intake)**:
  * **True Positives ($TP_{40}$)**: **7 fraud claims caught** (₹13,131 fraud rupees prevented).
  * **False Positives ($FP_{40}$)**: **73 genuine claims held** ($73 \times ₹380 = ₹27,740$ customer goodwill cost incurred).
  * **False Negatives ($FN_{40}$)**: **29 fraud claims missed** (auto-approved / unreviewed).
  * **True Negatives ($TN_{40}$)**: **1,313 genuine claims auto-approved / cleared without review**.
* **Under Default 0.5 Probability Threshold**:
  * Accuracy: **96.20%**
  * Precision: **15.38%**
  * Recall: **11.11%** (4 / 36 caught)
  * False Positive Rate: **1.59%** (22 genuine claims flagged out of 1,386)

---

## 2. Forensic Analysis of False Positives (High-Risk Genuine Claims)

We analyzed the highest-scoring non-fraud claims that were flagged in the top-40 queue.

### Top Highest-Risk Genuine Claims:
1. **`WC710496` (Score: 0.8328)**:
   * Claim Amount: ₹1,769 | Product: Induction Cooktop (List Price: ₹2,999) | Ratio: **0.5899**
   * Partner: Franchise in Bhopal | `partner_smoothed_fraud_rate`: **0.2051** (High prior fraud partner) | `partner_claims_prev_30d`: 2
   * **Root Cause**: The partner has a historically high fraud rate from 2021–2023. Combined with an above-average replacement ratio (59%), the model heavily penalizes the claim despite the physical inspection confirming a genuine failure.
2. **`WC710381` (Score: 0.8033)**:
   * Claim Amount: ₹1,181 | Product: Ceiling Fan (List Price: ₹1,999) | Ratio: **0.5908**
   * Partner: Franchise in Hubballi | `partner_smoothed_fraud_rate`: **0.1888**
   * **Root Cause**: Same dynamic — the partner's legacy reputation inflates the posterior probability.
3. **`WC711165` (Score: 0.7620)**:
   * Claim Amount: ₹8,237 | Product: Water Purifier (List Price: ₹11,999) | Ratio: **0.6865**
   * Partner: Franchise in Bhopal | `partner_smoothed_fraud_rate`: **0.1935**
   * **Root Cause**: Very high absolute claim amount (₹8,237) and 69% list price ratio filed by a historically tainted partner.
4. **`WC710554` (Score: 0.7037)**:
   * Claim Amount: ₹15,839 | Product: Robot Vacuum (List Price: ₹17,599) | Ratio: **0.9000**
   * Partner: Freelance Technician in Pune | `partner_smoothed_fraud_rate`: **0.1769**
   * **Root Cause**: Extreme replacement ratio (90% of list price) on a premium appliance.

### Systematic False Positive Patterns:
* **Partner Taint Overhang**: 80% of top false positives stem from 4 specific franchise partners in Bhopal, Hubballi, and Pune whose historical fraud rates in 2022–2024 were elevated. When these partners file valid claims, the model assigns high risk.
* **High Claim-to-List-Price Ratio**: Claims requesting $>60\%$ of the product list price are heavily penalized, even when legitimately requiring major sub-assembly replacement (e.g. PCB or motor swap).

---

## 3. Forensic Analysis of False Negatives (Missed Fraud Claims)

We analyzed the lowest-scoring confirmed fraud claims that escaped the top-40 queue.

### Top Lowest-Risk Missed Fraud Claims:
1. **`WC709956` (Score: 0.2263)**:
   * Claim Amount: **₹802** | Product: Air Fryer (List Price: ₹6,499) | Ratio: **0.1234**
   * Partner: Franchise in Warangal | `partner_smoothed_fraud_rate`: **0.0097** (Clean prior record) | `partner_claims_prev_30d`: 3
   * **Failure Mode**: The fraudster submitted a low-value claim (₹802) representing only 12% of product list price through a partner with zero historical fraud marks.
2. **`WC710711` (Score: 0.2189)**:
   * Claim Amount: **₹686** | Product: Mixer Grinder (List Price: ₹3,199) | Ratio: **0.2144**
   * Partner: Franchise in Jaipur | `partner_smoothed_fraud_rate`: **0.0048**
   * **Failure Mode**: Low claim amount, modest ratio, clean partner history.
3. **`WC709948` (Score: 0.1948)**:
   * Claim Amount: **₹681** | Product: Induction Cooktop (List Price: ₹4,048) | Ratio: **0.1682**
   * Partner: Franchise in Hubballi | `partner_smoothed_fraud_rate`: **0.0068**
   * **Failure Mode**: Minimal claim amount representing a routine minor repair.
4. **`WC710334` (Score: 0.1868)**:
   * Claim Amount: **₹984** | Product: Air Fryer (List Price: ₹8,773) | Ratio: **0.1122**
   * Partner: Authorised Service Centre in Pune | `partner_smoothed_fraud_rate`: **0.0256**
   * **Failure Mode**: Low ratio ($<12\%$) on a premium appliance.
5. **`WC710015` (Score: 0.1185)**:
   * Claim Amount: **₹1,603** | Product: Induction Cooktop (List Price: ₹4,048) | Ratio: **0.3960**
   * Partner: Authorised Service Centre in Pune | `partner_smoothed_fraud_rate`: **0.0053**
   * **Failure Mode**: Low partner historical fraud rate, average amount, standard fault description.

### Systematic False Negative Patterns:
* **The "Micro-Claim" Camouflage Strategy**:
  * In the post-May 1, 2026 regime, bad actors adapted to the auto-approval rule by submitting claims between **₹600 and ₹1,200** with low claim-to-list-price ratios ($<20\%$).
  * Because these claims mimic everyday low-cost repairs (e.g. gasket replacement, fuse change), their individual profile appears benign.
* **Partner Clean-Slate Concealment**:
  * 25 out of the 29 missed fraud claims were submitted by partners who had **0 confirmed fraud cases in the pre-May training period**.
  * Because their historical smoothed fraud rate was at or below baseline ($0.005–0.012$), the model had no historical evidence to flag them until volume burst signals accumulated.

---

## 4. Feature Sensitivity Across Error Classes

| Feature Dimension | True Positives ($TP_{40}$) | False Positives ($FP_{40}$) | False Negatives ($FN_{40}$) | True Negatives ($TN_{40}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Claim Amount** | ₹1,875.86 | ₹2,382.41 | ₹1,861.79 | ₹2,180.57 |
| **Claim / List Price Ratio** | **0.4682** | **0.5412** | **0.2829** | 0.2957 |
| **Partner Smoothed Fraud Rate** | **0.1084** | **0.0982** | **0.0426** | 0.0086 |
| **Partner Claims in Prev 30 Days** | **5.43** | 3.12 | 4.03 | 1.91 |
| **Sub-₹2,000 Claims in Prev 30 Days** | **4.71** | 2.21 | 3.17 | 1.39 |
| **Days Since Purchase** | 148 days | 192 days | 137 days | 241 days |
| **Customer Prior Claims** | 0.57 | 0.42 | 0.48 | 0.32 |

---

## 5. Actionable Recommendations for System Improvement

1. **Incorporate Partner-Level Burst Velocity Acceleration**:
   * Missed fraud claims come from partners whose volume jumped in May/June (`partner_claims_prev_30d` rose from 1 to 5). Enhancing features that measure the 14-day vs 60-day velocity ratio will help catch clean-slate partners during an active burst.
2. **Cluster Sub-₹2,000 Frequency by Partner**:
   * Flag partners who suddenly submit $>80\%$ of their monthly claims under ₹2,000, regardless of the individual claim amount.
3. **Desk Dynamic Capacity Recommendation**:
   * When fraud is concentrated in low-value claims (₹600–₹1,200), spending ₹380 to review a ₹680 claim offers marginal net recovery. The investigation desk should prioritize high-value borderline claims or partner-level audits rather than isolated micro-claims.
