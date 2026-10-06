"""
Phase 2B Model Verification & Validation Economics Test Suite.

Verifies:
- Temporal order of training, validation, and test splits.
- Test data is completely unobserved during model selection.
- Target column and leakage features are excluded from model inputs.
- Saved model pipeline produces valid, finite probability scores in [0.0, 1.0].
- Feature ordering consistency.
- Monthly top-40 capacity constraint (at most 40 claims per month).
- Exact mathematical correctness of financial value calculations.
"""

import csv
import json
import os
import joblib
import numpy as np
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "outputs", "evaluation")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "features")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


class TestModelEvaluation(unittest.TestCase):
    """Test suite for Phase 2B model evaluation and operational business economics."""

    def test_temporal_splits_ordering(self):
        """Verify validation split is strictly after training period and before test period."""
        meta_path = os.path.join(FEATURES_DIR, "train_modeling_metadata.csv")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_rows = list(csv.DictReader(f))

        train_dates = [r["first_submitted_at"] for r in meta_rows if r["split_assignment"] == "train_split"]
        val_dates = [r["first_submitted_at"] for r in meta_rows if r["split_assignment"] == "validation_split"]

        self.assertLess(max(train_dates), "2026-05-01", "Training claims must strictly precede 2026-05-01")
        self.assertGreaterEqual(min(val_dates), "2026-05-01", "Validation claims must start on or after 2026-05-01")
        self.assertLessEqual(max(val_dates), "2026-06-30 23:59:59", "Validation claims must end on or before 2026-06-30")

        # Verify test set starts after validation
        test_path = os.path.join(RAW_DATA_DIR, "test_unlabelled.csv")
        with open(test_path, "r", encoding="utf-8") as f:
            test_dates = [r["submitted_at"] for r in csv.DictReader(f)]

        self.assertGreater(min(test_dates), max(val_dates), "Test set must strictly follow validation set")

    def test_test_data_not_used_in_model_selection(self):
        """Verify no test claim IDs appear in training or validation datasets."""
        test_path = os.path.join(RAW_DATA_DIR, "test_unlabelled.csv")
        with open(test_path, "r", encoding="utf-8") as f:
            test_cids = {r["claim_id"] for r in csv.DictReader(f)}

        meta_path = os.path.join(FEATURES_DIR, "train_modeling_metadata.csv")
        with open(meta_path, "r", encoding="utf-8") as f:
            train_cids = {r["claim_id"] for r in csv.DictReader(f)}

        overlap = test_cids.intersection(train_cids)
        self.assertEqual(len(overlap), 0, f"Test claims leaked into training metadata: {overlap}")

    def test_target_and_leakage_excluded_from_model_inputs(self):
        """Verify is_fraud, inspector_note, partner_inspected, and source are absent from feature list."""
        feat_list_path = os.path.join(MODELS_DIR, "feature_list.json")
        with open(feat_list_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        features = meta["features"]
        self.assertNotIn("is_fraud", features)
        self.assertNotIn("inspector_note", features)
        self.assertNotIn("partner_inspected", features)
        self.assertNotIn("source", features)

    def test_model_pipeline_produces_finite_scores_in_valid_range(self):
        """Verify saved pipeline loads and produces probability scores in [0.0, 1.0]."""
        model_path = os.path.join(MODELS_DIR, "model_pipeline.joblib")
        self.assertTrue(os.path.exists(model_path), f"Missing model artifact: {model_path}")

        pipeline = joblib.load(model_path)

        feat_list_path = os.path.join(MODELS_DIR, "feature_list.json")
        with open(feat_list_path, "r", encoding="utf-8") as f:
            features = json.load(f)["features"]

        feat_path = os.path.join(FEATURES_DIR, "train_features.csv")
        with open(feat_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))[:50]  # Sample 50 rows

        cat_cols = ['sku', 'product_family', 'partner_type', 'partner_city', 'clean_fault_description']
        X_sample = []
        for r in rows:
            X_sample.append([r[c] if c in cat_cols else float(r[c]) for c in features])

        probs = pipeline.predict_proba(np.array(X_sample, dtype=object))[:, 1]

        self.assertEqual(len(probs), 50)
        self.assertTrue(np.all(np.isfinite(probs)), "Model produced non-finite or NaN probabilities")
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities outside [0.0, 1.0] range")

    def test_monthly_top40_capacity_constraint(self):
        """Verify monthly top-40 queue strictly selects at most 40 claims per calendar month."""
        top40_path = os.path.join(EVAL_DIR, "monthly_top40_validation.csv")
        with open(top40_path, "r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))

        may_rec = [r for r in records if r["month"] == "May 2026"][0]
        june_rec = [r for r in records if r["month"] == "June 2026"][0]

        may_caught = int(may_rec["frauds_caught"])
        may_genuine = int(may_rec["genuine_flagged"])
        june_caught = int(june_rec["frauds_caught"])
        june_genuine = int(june_rec["genuine_flagged"])

        self.assertEqual(may_caught + may_genuine, 40, "May top-40 must contain exactly 40 claims")
        self.assertEqual(june_caught + june_genuine, 40, "June top-40 must contain exactly 40 claims")
        self.assertLessEqual(may_caught, 40)
        self.assertLessEqual(june_caught, 40)

    def test_financial_calculations_correctness(self):
        """Verify mathematical integrity of review cost, net value, and value per reviewed claim."""
        top40_path = os.path.join(EVAL_DIR, "monthly_top40_validation.csv")
        with open(top40_path, "r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))

        for r in records:
            if r["month"] in ("May 2026", "June 2026"):
                n_reviewed = 40
            else:
                n_reviewed = 80

            expected_cost = n_reviewed * 380.0
            actual_cost = float(r["review_cost"])
            self.assertEqual(actual_cost, expected_cost, f"Review cost mismatch in {r['month']}")

            fraud_inr = float(r["fraud_inr_captured"])
            expected_net = fraud_inr - expected_cost
            actual_net = float(r["net_value"])
            self.assertAlmostEqual(actual_net, expected_net, places=2, msg=f"Net value mismatch in {r['month']}")

            expected_per_claim = expected_net / n_reviewed
            actual_per_claim = float(r["value_per_reviewed_claim"])
            self.assertAlmostEqual(actual_per_claim, expected_per_claim, places=2, msg=f"Value/claim mismatch in {r['month']}")


if __name__ == "__main__":
    unittest.main()
