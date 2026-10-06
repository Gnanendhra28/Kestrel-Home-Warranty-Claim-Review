"""
Phase 1 Data Integrity Tests for Kestrel Home Warranty Fraud Decision-Support System.
Verifies file presence, schema conformity, relational integrity, and immutability.
"""

import csv
import hashlib
import os
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
WORKSPACE_FILES_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "files")

# Expected SHA-256 hashes for source files
EXPECTED_HASHES = {
    "train.csv": "2490847a83be3e5613d206b4bd1e63b2a6e33887bbe101409013cf9cf02f2325",
    "test_unlabelled.csv": "2cd7f445c9216b4aedb77e31d650485687fefe37a2d3b7add654f3b51125bfd5",
    "partners.csv": "2a0ebd4cc4d43d65f4098e1180fa50c033fddaea9151f2f842216035cea594db",
    "products.csv": "649a753c5478c50713c0696ff02514a2a26b04a25a9a2c3eb4e5d50f6afa192b",
    "sample_submission.csv": "27df3b21f90423b3a1bf620d1e2646570ca872457078398ea91272d26eb9ba04",
    "ops-policy.pdf": "70e99583448d4ee5fbe10b0a0f1ff433d6fb74a453edda1d52f0fb8b4e1f7ecc",
    "email-thread.txt": "1927905ec21c17200d0033f01f3edbb67cfead4db34ce1617d1e42793108fbdc",
    "README.txt": "35500f6d939ff93238740ca28452fd52ac29e02c4c5b95c275a412685469f9d0",
}

EXPECTED_TRAIN_COLUMNS = [
    "claim_id",
    "submitted_at",
    "partner_id",
    "sku",
    "product_serial",
    "days_since_purchase",
    "claim_amount_inr",
    "photo_attached",
    "partner_inspected",
    "claim_description",
    "inspector_note",
    "customer_prior_claims",
    "source",
    "is_fraud",
]

EXPECTED_TEST_COLUMNS = [c for c in EXPECTED_TRAIN_COLUMNS if c != "is_fraud"]


def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestDataIntegrity(unittest.TestCase):
    """Basic forensic data-integrity assertions."""

    def test_required_raw_files_exist(self):
        """Verify all canonical files exist in data/raw."""
        for filename in EXPECTED_HASHES:
            path = os.path.join(RAW_DATA_DIR, filename)
            self.assertTrue(
                os.path.exists(path), f"Required file missing: {path}"
            )

    def test_source_files_hashes_unmodified(self):
        """Verify byte-level immutability of raw files against recorded hashes."""
        for filename, expected_hash in EXPECTED_HASHES.items():
            path = os.path.join(RAW_DATA_DIR, filename)
            actual_hash = compute_sha256(path)
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"File hash mismatch for {filename}! File was modified.",
            )

    def test_train_schema_and_row_count(self):
        """Verify train.csv column schema, row count, and claim_id column."""
        path = os.path.join(RAW_DATA_DIR, "train.csv")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        self.assertEqual(header, EXPECTED_TRAIN_COLUMNS)
        self.assertEqual(len(rows), 12029)
        self.assertIn("claim_id", header)
        self.assertIn("partner_id", header)
        self.assertIn("sku", header)
        self.assertIn("is_fraud", header)

    def test_test_schema_and_row_count(self):
        """Verify test_unlabelled.csv schema, row count, and lack of target."""
        path = os.path.join(RAW_DATA_DIR, "test_unlabelled.csv")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        self.assertEqual(header, EXPECTED_TEST_COLUMNS)
        self.assertEqual(len(rows), 2252)
        self.assertNotIn("is_fraud", header)

    def test_sample_submission_integrity(self):
        """Verify sample_submission.csv format, row count, and claim_id alignment."""
        sub_path = os.path.join(RAW_DATA_DIR, "sample_submission.csv")
        test_path = os.path.join(RAW_DATA_DIR, "test_unlabelled.csv")

        with open(sub_path, "r", encoding="utf-8") as f:
            sub_reader = csv.DictReader(f)
            self.assertEqual(sub_reader.fieldnames, ["claim_id", "score"])
            sub_rows = list(sub_reader)

        with open(test_path, "r", encoding="utf-8") as f:
            test_reader = csv.DictReader(f)
            test_rows = list(test_reader)

        self.assertEqual(len(sub_rows), 2252)
        sub_claim_ids = [r["claim_id"] for r in sub_rows]
        test_claim_ids = [r["claim_id"] for r in test_rows]

        self.assertEqual(
            sub_claim_ids,
            test_claim_ids,
            "Sample submission claim_ids do not match test_unlabelled.csv exactly in order!",
        )

    def test_partner_and_sku_referential_integrity(self):
        """Verify all partner_id and sku references map into partners.csv and products.csv."""
        partners_path = os.path.join(RAW_DATA_DIR, "partners.csv")
        products_path = os.path.join(RAW_DATA_DIR, "products.csv")
        train_path = os.path.join(RAW_DATA_DIR, "train.csv")
        test_path = os.path.join(RAW_DATA_DIR, "test_unlabelled.csv")

        with open(partners_path, "r", encoding="utf-8") as f:
            valid_partners = {r["partner_id"] for r in csv.DictReader(f)}

        with open(products_path, "r", encoding="utf-8") as f:
            valid_skus = {r["sku"] for r in csv.DictReader(f)}

        self.assertEqual(len(valid_partners), 380)
        self.assertEqual(len(valid_skus), 21)

        for path in (train_path, test_path):
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.assertIn(
                        row["partner_id"],
                        valid_partners,
                        f"Unknown partner {row['partner_id']} in {path}",
                    )
                    self.assertIn(
                        row["sku"],
                        valid_skus,
                        f"Unknown SKU {row['sku']} in {path}",
                    )


if __name__ == "__main__":
    unittest.main()
