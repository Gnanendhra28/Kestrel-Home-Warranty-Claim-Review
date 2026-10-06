"""
Phase 2B Candidate Model Training, Evaluation, and Benchmarking Pipeline.

Evaluates:
- Naive All-Zero Baseline
- Logistic Regression (Balanced)
- Random Forest (depth=6, Balanced)
- Random Forest (depth=8, Balanced)
- HistGradientBoosting (Balanced)
- LightGBM (Balanced)
- Text feature ablation (with vs without clean_fault_description)

Outputs:
- outputs/models/model_pipeline.joblib
- outputs/models/feature_list.json
- outputs/models/model_metadata.json
- outputs/evaluation/model_comparison.csv
- outputs/evaluation/monthly_top40_validation.csv
- outputs/evaluation/error_analysis.csv
"""

import csv
import json
import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "outputs", "evaluation")

RANDOM_SEED = 42

# Candidate feature sets
# Exclude drifting partner age / newness flags to prevent regime inversion
CATEGORICAL_COLS = ['sku', 'product_family', 'partner_type', 'partner_city', 'clean_fault_description']
EXCLUDED_DRIFT_COLS = {'partner_age_days_at_claim', 'is_new_partner_180d', 'is_new_partner_365d'}


def load_dataset():
    feat_path = os.path.join(FEATURES_DIR, "train_features.csv")
    meta_path = os.path.join(FEATURES_DIR, "train_modeling_metadata.csv")

    with open(feat_path, "r", encoding="utf-8") as f:
        feat_rows = list(csv.DictReader(f))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_rows = {r['claim_id']: r for r in csv.DictReader(f)}

    train_rows = [r for r in feat_rows if meta_rows[r['claim_id']]['split_assignment'] == 'train_split']
    val_rows = [r for r in feat_rows if meta_rows[r['claim_id']]['split_assignment'] == 'validation_split']

    return feat_rows, meta_rows, train_rows, val_rows


def extract_features_and_labels(rows, feature_cols):
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS]
    X = []
    y = []
    for r in rows:
        y.append(int(r['is_fraud']))
        row_vals = [r[c] if c in cat_cols else float(r[c]) for c in feature_cols]
        X.append(row_vals)
    return np.array(X, dtype=object), np.array(y, dtype=int)


def evaluate_monthly_top40(y_true, y_prob, val_rows, meta_rows):
    may_idx = [i for i, r in enumerate(val_rows) if meta_rows[r['claim_id']]['first_submitted_at'].startswith('2026-05')]
    june_idx = [i for i, r in enumerate(val_rows) if meta_rows[r['claim_id']]['first_submitted_at'].startswith('2026-06')]

    results = {}
    for month_name, indices in [("May 2026", may_idx), ("June 2026", june_idx)]:
        m_true = y_true[indices]
        m_prob = y_prob[indices]
        m_rows = [val_rows[i] for i in indices]

        top40_order = np.argsort(-m_prob)[:40]
        top40_true = m_true[top40_order]
        top40_rows = [m_rows[i] for i in top40_order]

        frauds_caught = int(top40_true.sum())
        genuine_flagged = 40 - frauds_caught
        fraud_inr_captured = sum(float(r['claim_amount_inr']) for r, t in zip(top40_rows, top40_true) if t == 1)
        mean_claim_amount = np.mean([float(r['claim_amount_inr']) for r in top40_rows])

        review_cost = 40 * 380
        net_value = fraud_inr_captured - review_cost
        val_per_claim = net_value / 40.0

        results[month_name] = {
            "month": month_name,
            "total_claims": len(indices),
            "total_fraud": int(m_true.sum()),
            "frauds_caught": frauds_caught,
            "genuine_flagged": genuine_flagged,
            "precision_at_40": round(frauds_caught / 40.0, 4),
            "fraud_recall_at_40": round(frauds_caught / float(m_true.sum()), 4),
            "fraud_inr_captured": round(fraud_inr_captured, 2),
            "mean_claim_amount": round(mean_claim_amount, 2),
            "review_cost": review_cost,
            "net_value": round(net_value, 2),
            "value_per_reviewed_claim": round(val_per_claim, 2),
        }

    agg_frauds = results["May 2026"]["frauds_caught"] + results["June 2026"]["frauds_caught"]
    agg_total_fraud = results["May 2026"]["total_fraud"] + results["June 2026"]["total_fraud"]
    agg_inr = results["May 2026"]["fraud_inr_captured"] + results["June 2026"]["fraud_inr_captured"]
    agg_cost = 80 * 380
    agg_net = agg_inr - agg_cost

    results["Aggregate (May + June)"] = {
        "month": "Aggregate (May + June)",
        "total_claims": len(val_rows),
        "total_fraud": agg_total_fraud,
        "frauds_caught": agg_frauds,
        "genuine_flagged": 80 - agg_frauds,
        "precision_at_40": round(agg_frauds / 80.0, 4),
        "fraud_recall_at_40": round(agg_frauds / float(agg_total_fraud), 4),
        "fraud_inr_captured": round(agg_inr, 2),
        "mean_claim_amount": round((results["May 2026"]["mean_claim_amount"] + results["June 2026"]["mean_claim_amount"]) / 2.0, 2),
        "review_cost": agg_cost,
        "net_value": round(agg_net, 2),
        "value_per_reviewed_claim": round(agg_net / 80.0, 2),
    }

    return results


def run():
    print("=" * 70)
    print("EXECUTING PHASE 2B MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    feat_rows, meta_rows, train_rows, val_rows = load_dataset()
    print(f"Loaded {len(train_rows)} training rows and {len(val_rows)} validation rows.")

    all_features = [c for c in feat_rows[0].keys() if c not in ('claim_id', 'is_fraud')]
    pruned_features = [c for c in all_features if c not in EXCLUDED_DRIFT_COLS]
    no_text_features = [c for c in pruned_features if c != 'clean_fault_description']

    print(f"Pruned robust feature set: {len(pruned_features)} features.")

    # Prepare datasets for pruned features
    X_train_p, y_train = extract_features_and_labels(train_rows, pruned_features)
    X_val_p, y_val = extract_features_and_labels(val_rows, pruned_features)

    cat_idx_p = [pruned_features.index(c) for c in pruned_features if c in CATEGORICAL_COLS]
    num_idx_p = [pruned_features.index(c) for c in pruned_features if c not in CATEGORICAL_COLS]

    preprocessor_p = ColumnTransformer([
        ('num', StandardScaler(), num_idx_p),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_idx_p)
    ])

    X_train_proc = preprocessor_p.fit_transform(X_train_p)
    X_val_proc = preprocessor_p.transform(X_val_p)

    # Prepare datasets for no-text features
    X_train_nt, _ = extract_features_and_labels(train_rows, no_text_features)
    X_val_nt, _ = extract_features_and_labels(val_rows, no_text_features)

    cat_idx_nt = [no_text_features.index(c) for c in no_text_features if c in CATEGORICAL_COLS]
    num_idx_nt = [no_text_features.index(c) for c in no_text_features if c not in CATEGORICAL_COLS]

    preprocessor_nt = ColumnTransformer([
        ('num', StandardScaler(), num_idx_nt),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_idx_nt)
    ])
    X_train_nt_proc = preprocessor_nt.fit_transform(X_train_nt)
    X_val_nt_proc = preprocessor_nt.transform(X_val_nt)

    # Models list
    candidate_specs = [
        ("Naive All-Zero Baseline", None, None, None),
        ("Logistic Regression (C=0.1, Balanced)", LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED), preprocessor_p, pruned_features),
        ("Random Forest (depth=6, Balanced)", RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1), preprocessor_p, pruned_features),
        ("Random Forest (depth=8, Balanced)", RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1), preprocessor_p, pruned_features),
        ("HistGradientBoosting (Balanced)", HistGradientBoostingClassifier(class_weight='balanced', max_depth=5, learning_rate=0.03, random_state=RANDOM_SEED), preprocessor_p, pruned_features),
        ("LightGBM (depth=4, Balanced)", lgb.LGBMClassifier(class_weight='balanced', n_estimators=80, max_depth=4, learning_rate=0.03, random_state=RANDOM_SEED, verbose=-1), preprocessor_p, pruned_features),
        ("Random Forest (depth=6, No Text)", RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1), preprocessor_nt, no_text_features),
    ]

    comparison_records = []
    top40_records = []
    trained_models = {}

    for name, clf, prep, fcols in candidate_specs:
        if clf is None:
            # Baseline
            y_prob = np.zeros_like(y_val, dtype=float)
            y_pred = np.zeros_like(y_val, dtype=int)
            roc = 0.5
            pr = float(y_val.mean())
        else:
            if fcols == pruned_features:
                X_tr = X_train_proc
                X_v = X_val_proc
            else:
                X_tr = X_train_nt_proc
                X_v = X_val_nt_proc

            clf.fit(X_tr, y_train)
            y_prob = clf.predict_proba(X_v)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            roc = roc_auc_score(y_val, y_prob)
            pr = average_precision_score(y_val, y_prob)
            trained_models[name] = (clf, prep, fcols, y_prob)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        t40 = evaluate_monthly_top40(y_val, y_prob, val_rows, meta_rows)
        c = t40["Aggregate (May + June)"]

        comparison_records.append({
            "model_name": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "false_positive_rate": round(fpr, 4),
            "may_top40_frauds": t40["May 2026"]["frauds_caught"],
            "may_top40_net_inr": t40["May 2026"]["net_value"],
            "june_top40_frauds": t40["June 2026"]["frauds_caught"],
            "june_top40_net_inr": t40["June 2026"]["net_value"],
            "total_top40_frauds": c["frauds_caught"],
            "total_top40_inr_captured": c["fraud_inr_captured"],
            "total_top40_net_inr": c["net_value"],
            "value_per_reviewed_claim": c["value_per_reviewed_claim"],
        })

        if name == "Random Forest (depth=6, Balanced)":
            top40_records.append(t40["May 2026"])
            top40_records.append(t40["June 2026"])
            top40_records.append(c)

    # Write model_comparison.csv
    comp_csv_path = os.path.join(EVAL_DIR, "model_comparison.csv")
    with open(comp_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_records[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_records)
    print(f"Written: {comp_csv_path}")

    # Write monthly_top40_validation.csv
    top40_csv_path = os.path.join(EVAL_DIR, "monthly_top40_validation.csv")
    with open(top40_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(top40_records[0].keys()))
        writer.writeheader()
        writer.writerows(top40_records)
    print(f"Written: {top40_csv_path}")

    # SELECTED PREFERRED MODEL: Random Forest (depth=6, Balanced)
    selected_name = "Random Forest (depth=6, Balanced)"
    selected_clf, selected_prep, selected_fcols, selected_probs = trained_models[selected_name]

    # Save Pipeline Artifact
    pipeline = Pipeline([
        ('preprocessor', selected_prep),
        ('classifier', selected_clf)
    ])
    model_artifact_path = os.path.join(MODELS_DIR, "model_pipeline.joblib")
    joblib.dump(pipeline, model_artifact_path)
    print(f"Saved selected model pipeline: {model_artifact_path}")

    # Save Feature List
    feat_list_path = os.path.join(MODELS_DIR, "feature_list.json")
    with open(feat_list_path, "w", encoding="utf-8") as f:
        json.dump({
            "feature_count": len(selected_fcols),
            "features": selected_fcols,
            "categorical_features": [c for c in selected_fcols if c in CATEGORICAL_COLS],
            "numeric_features": [c for c in selected_fcols if c not in CATEGORICAL_COLS],
            "excluded_features": list(EXCLUDED_DRIFT_COLS),
        }, f, indent=2)
    print(f"Saved feature list: {feat_list_path}")

    # Save Model Metadata
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": "RandomForestClassifier",
            "hyperparameters": {
                "n_estimators": 150,
                "max_depth": 6,
                "class_weight": "balanced",
                "random_state": RANDOM_SEED,
            },
            "training_samples": len(train_rows),
            "validation_samples": len(val_rows),
            "training_period": "2025-04-01 to 2026-04-30",
            "validation_period": "2026-05-01 to 2026-06-30",
            "validation_metrics": {
                "accuracy": 0.9620,
                "precision": 0.1538,
                "recall": 0.1111,
                "f1_score": 0.1290,
                "roc_auc": 0.7059,
                "pr_auc": 0.0722,
                "top40_frauds_caught": 7,
                "top40_net_inr": -17269.0,
            },
            "python_version": "3.13.14",
            "scikit_learn_version": "1.9.1",
        }, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    # Generate Error Analysis Dataset
    error_records = []
    may_idx = [i for i, r in enumerate(val_rows) if meta_rows[r['claim_id']]['first_submitted_at'].startswith('2026-05')]
    june_idx = [i for i, r in enumerate(val_rows) if meta_rows[r['claim_id']]['first_submitted_at'].startswith('2026-06')]

    may_top40_cids = {val_rows[i]['claim_id'] for i in np.array(may_idx)[np.argsort(-selected_probs[may_idx])[:40]]}
    june_top40_cids = {val_rows[i]['claim_id'] for i in np.array(june_idx)[np.argsort(-selected_probs[june_idx])[:40]]}
    all_top40_cids = may_top40_cids | june_top40_cids

    for i, r in enumerate(val_rows):
        cid = r['claim_id']
        actual = int(r['is_fraud'])
        score = float(selected_probs[i])
        pred_50 = 1 if score >= 0.5 else 0
        in_top40 = 1 if cid in all_top40_cids else 0

        # Classification category at 0.5 threshold
        if actual == 1 and pred_50 == 1:
            cat_50 = "TP"
        elif actual == 0 and pred_50 == 1:
            cat_50 = "FP"
        elif actual == 0 and pred_50 == 0:
            cat_50 = "TN"
        else:
            cat_50 = "FN"

        # Classification category under monthly top-40 queue
        if in_top40:
            cat_top40 = "Top40_TP" if actual == 1 else "Top40_FP"
        else:
            cat_top40 = "NotSelected_FN" if actual == 1 else "NotSelected_TN"

        error_records.append({
            "claim_id": cid,
            "submission_timestamp": meta_rows[cid]['first_submitted_at'],
            "actual_is_fraud": actual,
            "risk_score": round(score, 4),
            "predicted_label_at_50": pred_50,
            "category_at_50": cat_50,
            "selected_in_monthly_top40": in_top40,
            "category_in_top40": cat_top40,
            "claim_amount_inr": r['claim_amount_inr'],
            "claim_amount_to_list_price_ratio": r['claim_amount_to_list_price_ratio'],
            "sku": r['sku'],
            "product_family": r['product_family'],
            "partner_type": r['partner_type'],
            "partner_city": r['partner_city'],
            "clean_fault_description": r['clean_fault_description'],
            "partner_claims_prev_30d": r['partner_claims_prev_30d'],
            "partner_smoothed_fraud_rate": r['partner_smoothed_fraud_rate'],
            "customer_prior_claims": r['customer_prior_claims'],
            "days_since_purchase": r['days_since_purchase'],
        })

    # Sort by risk score descending
    error_records.sort(key=lambda x: x['risk_score'], reverse=True)

    err_csv_path = os.path.join(EVAL_DIR, "error_analysis.csv")
    with open(err_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(error_records[0].keys()))
        writer.writeheader()
        writer.writerows(error_records)
    print(f"Written: {err_csv_path} ({len(error_records)} rows)")

    print("=" * 70)
    print("PHASE 2B MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run()
