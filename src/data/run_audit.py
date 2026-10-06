"""
Phase 1 Forensic Audit CLI Script for Kestrel Home Warranty Fraud.
Executes read-only statistical checks and prints key audit summaries.
"""

import csv
import collections
from datetime import datetime
import os
import statistics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def load_csv(filename):
    path = os.path.join(RAW_DATA_DIR, filename)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def run():
    print("=" * 70)
    print("KESTREL HOME — PHASE 1 FORENSIC AUDIT SUMMARY")
    print("=" * 70)

    train_fields, train = load_csv("train.csv")
    test_fields, test = load_csv("test_unlabelled.csv")
    partners_fields, partners = load_csv("partners.csv")
    products_fields, products = load_csv("products.csv")
    sub_fields, sub = load_csv("sample_submission.csv")

    print(f"Train dataset:     {len(train):6d} rows, {len(train_fields)} columns")
    print(f"Test dataset:      {len(test):6d} rows, {len(test_fields)} columns")
    print(f"Partners catalog:  {len(partners):6d} rows, {len(partners_fields)} columns")
    print(f"Products catalog:  {len(products):6d} rows, {len(products_fields)} columns")
    print(f"Sample submission: {len(sub):6d} rows, {len(sub_fields)} columns")

    # Target distribution
    target_counts = collections.Counter(r["is_fraud"] for r in train)
    print("\n--- Target Distribution (train.csv) ---")
    print(f"  Non-fraud (0): {target_counts['0']:5d} ({target_counts['0']/len(train)*100:.2f}%)")
    print(f"  Fraud (1):     {target_counts['1']:5d} ({target_counts['1']/len(train)*100:.2f}%)")
    print(f"  Undecided ('') {target_counts['']:5d} ({target_counts['']/len(train)*100:.2f}%)")

    # Duplicates
    train_cids = collections.Counter(r["claim_id"] for r in train)
    dup_tr = sum(1 for v in train_cids.values() if v > 1)
    test_cids = collections.Counter(r["claim_id"] for r in test)
    dup_te = sum(1 for v in test_cids.values() if v > 1)
    print("\n--- Claim ID Duplication ---")
    print(f"  Train duplicated claim_ids: {dup_tr} (total duplicate rows: {len(train) - len(train_cids)})")
    print(f"  Test duplicated claim_ids:  {dup_te}")

    # Dates
    tr_dates = [datetime.fromisoformat(r["submitted_at"].replace("Z", "")) for r in train]
    te_dates = [datetime.fromisoformat(r["submitted_at"].replace("Z", "")) for r in test]
    print("\n--- Temporal Horizons ---")
    print(f"  Train: {min(tr_dates)} to {max(tr_dates)} (15 months)")
    print(f"  Test:  {min(te_dates)} to {max(te_dates)} (3 months, Q3 2026)")

    # Policy 1 May 2026 impact
    under_2k_tr = sum(1 for r in train if float(r["claim_amount_inr"]) < 2000)
    under_2k_te = sum(1 for r in test if float(r["claim_amount_inr"]) < 2000)
    print("\n--- Policy Impact (< ₹2,000 Auto-Approval) ---")
    print(f"  Train sub-₹2,000 claims: {under_2k_tr} ({under_2k_tr/len(train)*100:.1f}%)")
    print(f"  Test sub-₹2,000 claims:  {under_2k_te} ({under_2k_te/len(test)*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run()
