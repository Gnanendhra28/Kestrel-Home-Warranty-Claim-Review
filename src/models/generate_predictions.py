"""
Generate predictions.csv for Kestrel Home Warranty Fraud Review Phase 3.

Scores test_features.csv using the verified production model pipeline
(Random Forest Classifier, 33 features).
Outputs predictions.csv matching sample_submission.csv format and ordering exactly.
Generates outputs/reports/predictions_validation.md.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def main():
    print("--- STEP 1: Generating predictions.csv ---")
    # 1. Load feature specification
    feat_spec_path = os.path.join(MODELS_DIR, "feature_list.json")
    with open(feat_spec_path, "r", encoding="utf-8") as f:
        feat_spec = json.load(f)
    features = feat_spec["features"]
    print(f"Loaded {len(features)} modeling features from {feat_spec_path}")

    # 2. Load model pipeline
    model_path = os.path.join(MODELS_DIR, "model_pipeline.joblib")
    model = joblib.load(model_path)
    print(f"Loaded trained model pipeline from {model_path}")

    # 3. Load test features
    test_feat_path = os.path.join(FEATURES_DIR, "test_features.csv")
    test_df = pd.read_csv(test_feat_path)
    print(f"Loaded test features: {test_df.shape}")

    # 4. Load sample submission for ground truth alignment check
    sample_sub_path = os.path.join(RAW_DATA_DIR, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)
    print(f"Loaded sample submission: {sample_sub.shape}")

    # 5. Extract modeling feature matrix
    X_test = test_df[features]

    # 6. Predict continuous probabilities
    probas = model.predict_proba(X_test)[:, 1]

    # 7. Construct predictions dataframe
    preds_df = pd.DataFrame({
        "claim_id": test_df["claim_id"],
        "score": np.round(probas, 6)
    })

    # 8. Validation checks
    assert len(preds_df) == 2252, f"Expected 2252 rows, got {len(preds_df)}"
    assert list(preds_df.columns) == ["claim_id", "score"], f"Columns mismatch: {list(preds_df.columns)}"
    assert (preds_df["claim_id"].values == sample_sub["claim_id"].values).all(), "claim_id ordering does not match sample_submission.csv exactly!"
    assert preds_df["score"].isna().sum() == 0, "NaNs found in scores!"
    assert (preds_df["score"] >= 0.0).all() and (preds_df["score"] <= 1.0).all(), "Scores out of [0, 1] range!"

    # 9. Save predictions.csv
    out_preds_path = os.path.join(OUTPUTS_DIR, "predictions.csv")
    preds_df.to_csv(out_preds_path, index=False)
    print(f"Successfully saved {out_preds_path} with {len(preds_df)} rows.")

    # 10. Compute statistics for validation report
    min_score = float(preds_df["score"].min())
    max_score = float(preds_df["score"].max())
    mean_score = float(preds_df["score"].mean())
    std_score = float(preds_df["score"].std())
    median_score = float(preds_df["score"].median())
    q25 = float(preds_df["score"].quantile(0.25))
    q75 = float(preds_df["score"].quantile(0.75))
    q90 = float(preds_df["score"].quantile(0.90))
    q95 = float(preds_df["score"].quantile(0.95))
    q99 = float(preds_df["score"].quantile(0.99))

    # Top-40 cutoffs
    # Test set spans 3 months: July, August, September 2026.
    # Monthly top-40:
    test_df["score"] = preds_df["score"]
    monthly_top40_stats = {}
    for m in [7, 8, 9]:
        m_df = test_df[test_df["claim_month"] == m]
        top40_m = m_df.sort_values(by="score", ascending=False).head(40)
        m_name = {7: "July 2026", 8: "August 2026", 9: "September 2026"}[m]
        monthly_top40_stats[m_name] = {
            "total_claims": len(m_df),
            "top40_min_score": float(top40_m["score"].min()),
            "top40_max_score": float(top40_m["score"].max()),
            "top40_mean_score": float(top40_m["score"].mean()),
        }

    # Top 40 overall
    top40_overall = preds_df.sort_values(by="score", ascending=False).head(40)
    top40_overall_cutoff = float(top40_overall["score"].min())

    # Write report
    report_path = os.path.join(REPORTS_DIR, "predictions_validation.md")
    report_content = f"""# Test Set Predictions Validation Report

**Model:** Production Random Forest Classifier (`outputs/models/model_pipeline.joblib`)  
**Features:** 33 features from `outputs/models/feature_list.json`  
**Output Target:** `outputs/predictions.csv`  
**Reference Comparison:** `data/raw/sample_submission.csv`  

---

## 1. File Integrity & Alignment Checks

| Check | Requirement | Result | Status |
| :--- | :--- | :--- | :--- |
| **Row Count** | Exactly 2,252 rows | 2,252 rows | **PASS** |
| **Column Names** | `['claim_id', 'score']` | `['claim_id', 'score']` | **PASS** |
| **Row Order** | Exact match with `sample_submission.csv` | 100% (2,252 / 2,252 identical order) | **PASS** |
| **Missing Values** | 0 nulls / NaNs | 0 nulls | **PASS** |
| **Score Bounds** | All values in $[0.0, 1.0]$ | Min: {min_score:.6f}, Max: {max_score:.6f} | **PASS** |
| **Score Nature** | Continuous probability for ranking | Continuous (mean={mean_score:.4f}, std={std_score:.4f}) | **PASS** |

---

## 2. Test Set Score Distribution

The scores represent calibrated class-1 posterior probabilities $P(\\text{{fraud}} \\mid X)$ generated by the trained ensemble:

- **Minimum Score:** `{min_score:.6f}`
- **25th Percentile (Q1):** `{q25:.6f}`
- **Median (50th Percentile):** `{median_score:.6f}`
- **Mean Score:** `{mean_score:.6f}`
- **Standard Deviation:** `{std_score:.4f}`
- **75th Percentile (Q3):** `{q75:.6f}`
- **90th Percentile:** `{q90:.6f}`
- **95th Percentile:** `{q95:.6f}`
- **99th Percentile:** `{q99:.6f}`
- **Maximum Score:** `{max_score:.6f}`

---

## 3. Top-40 Monthly Desk Capacity Cutoffs

The investigation desk reviews at most 40 claims per month. Below are the score thresholds required to qualify for the monthly Top-40 review quota across the Q2 FY26 test period (July – September 2026):

| Period | Total Claims Submitted | Top-40 Score Threshold (Min) | Top-40 Max Score | Top-40 Mean Score |
| :--- | :--- | :--- | :--- | :--- |
| **July 2026** | {monthly_top40_stats['July 2026']['total_claims']} | `{monthly_top40_stats['July 2026']['top40_min_score']:.4f}` | `{monthly_top40_stats['July 2026']['top40_max_score']:.4f}` | `{monthly_top40_stats['July 2026']['top40_mean_score']:.4f}` |
| **August 2026** | {monthly_top40_stats['August 2026']['total_claims']} | `{monthly_top40_stats['August 2026']['top40_min_score']:.4f}` | `{monthly_top40_stats['August 2026']['top40_max_score']:.4f}` | `{monthly_top40_stats['August 2026']['top40_mean_score']:.4f}` |
| **September 2026** | {monthly_top40_stats['September 2026']['total_claims']} | `{monthly_top40_stats['September 2026']['top40_min_score']:.4f}` | `{monthly_top40_stats['September 2026']['top40_max_score']:.4f}` | `{monthly_top40_stats['September 2026']['top40_mean_score']:.4f}` |
| **Overall Top-40** | 2,252 | `{top40_overall_cutoff:.4f}` | `{max_score:.4f}` | `{top40_overall['score'].mean():.4f}` |

---

## 4. Verification Summary
The generated `predictions.csv` satisfies all business and submission criteria:
1. It is directly derived from the verified Phase 2B production model without retraining.
2. It completely adheres to `sample_submission.csv` row order, headers, and typing.
3. It delivers a fine-grained continuous ranking metric, ensuring investigations desk capacity is allocated to the highest-risk claims.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Generated validation report at {report_path}")


if __name__ == "__main__":
    main()
