"""
Phase 2A Leakage-Safe Feature Engineering Pipeline for Kestrel Home Warranty Fraud.

Implements:
- Canonical claim deduplication (earliest submission timestamp).
- Temporal index building for partner volume, burst, and Bayesian-smoothed fraud rates.
- Serial recycling detection.
- Domain-specific product wear and warranty ratios.
- Sanitized fault category classification (stripping adversarial prompt injections).
- Policy-shift indicators (1 May 2026 sub-₹2,000 auto-approval rule).
- Strict exclusion of operational leakage (inspector_note, partner_inspected, future labels).
"""

import bisect
import csv
import collections
from datetime import datetime, timedelta
import math
import os
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
FEATURES_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "features")

CANONICAL_FAULTS = [
    "unit not heating",
    "motor not running",
    "tripping mcb",
    "filter indicator stuck",
    "remote not working",
    "blade jammed",
    "loud noise while running",
    "power button not working",
    "display not working",
    "water leaking",
    "burning smell",
    "not charging",
]

# Production-safe feature definitions (ordered list of column names)
PRODUCTION_FEATURE_COLUMNS = [
    # Claim basics
    "claim_amount_inr",
    "log_claim_amount",
    "is_sub_2000",
    "claim_amount_threshold_gap",
    "photo_attached",
    "customer_prior_claims",
    # Product & Warranty
    "sku",
    "product_family",
    "product_list_price",
    "product_warranty_months",
    "claim_amount_to_list_price_ratio",
    "days_since_purchase",
    "days_to_warranty_ratio",
    "is_near_warranty_expiry",
    "is_early_life_claim",
    # Partner Profile
    "partner_type",
    "partner_city",
    "partner_age_days_at_claim",
    "is_new_partner_180d",
    "is_new_partner_365d",
    # Partner History & Velocity (strictly prior to claim)
    "partner_claims_prev_7d",
    "partner_claims_prev_30d",
    "partner_claims_lifetime_prior",
    "partner_claim_velocity_ratio",
    "partner_sub2000_claims_prev_30d",
    "partner_sub2000_ratio_prev_30d",
    # Target-Based Partner Features (strictly prior, Bayesian smoothed)
    "partner_prior_fraud_count",
    "partner_prior_resolved_claims",
    "partner_smoothed_fraud_rate",
    # Serial History (strictly prior to claim)
    "serial_prior_claim_count",
    # Calendar & Policy
    "claim_month",
    "claim_day_of_week",
    "claim_hour",
    "is_weekend",
    "post_policy_change",
    # Sanitized Fault Description
    "clean_fault_description",
]


def load_csv(filename: str) -> Tuple[List[str], List[Dict[str, str]]]:
    path = os.path.join(RAW_DATA_DIR, filename)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def clean_fault_description(text: str) -> str:
    """
    Sanitizes partner fault description by mapping to 12 canonical fault types.
    Neutralizes adversarial prompt injections embedded in raw free-text fields.
    """
    if not text:
        return "other"
    text_lower = text.lower().strip()
    if "display blank" in text_lower:
        return "display not working"
    for fault in CANONICAL_FAULTS:
        if fault in text_lower:
            return fault
    return "other"


class TemporalHistoryIndex:
    """
    Maintains chronological event logs to compute strictly prior aggregates.
    Guarantees no future leakage and excludes current claim from historical metrics.
    """

    def __init__(self, global_prior_rate: float = 0.01265, smoothing_weight: float = 20.0):
        self.global_prior_rate = global_prior_rate
        self.smoothing_weight = smoothing_weight
        # partner_id -> sorted list of (timestamp, is_sub_2000)
        self.partner_events: Dict[str, List[Tuple[datetime, bool]]] = collections.defaultdict(list)
        # partner_id -> sorted list of (timestamp, is_fraud_int) for resolved training claims
        self.partner_target_events: Dict[str, List[Tuple[datetime, int]]] = collections.defaultdict(list)
        # serial -> sorted list of timestamps
        self.serial_events: Dict[str, List[datetime]] = collections.defaultdict(list)

    def populate(self, all_claims: List[Dict[str, Any]], label_cutoff: datetime):
        """
        Indexes claims for fast bisection. Only claims submitted strictly before
        label_cutoff and possessing a resolved target ('0' or '1') contribute to target events.
        """
        for r in all_claims:
            t = r["submitted_dt"]
            pid = r["partner_id"]
            serial = r["product_serial"]
            is_sub2k = float(r["claim_amount_inr"]) < 2000.0

            self.partner_events[pid].append((t, is_sub2k))
            self.serial_events[serial].append(t)

            # Target events only from resolved training claims before label_cutoff
            if t < label_cutoff and r.get("is_fraud") in ("0", "1"):
                is_fraud_val = int(r["is_fraud"])
                self.partner_target_events[pid].append((t, is_fraud_val))

        # Ensure lists are strictly sorted by timestamp
        for pid in self.partner_events:
            self.partner_events[pid].sort(key=lambda x: x[0])
        for pid in self.partner_target_events:
            self.partner_target_events[pid].sort(key=lambda x: x[0])
        for serial in self.serial_events:
            self.serial_events[serial].sort()

    def query_partner_metrics(self, pid: str, claim_dt: datetime) -> Dict[str, float]:
        """
        Extracts partner history strictly prior to claim_dt ([T_prior < claim_dt]).
        """
        events = self.partner_events.get(pid, [])
        timestamps = [e[0] for e in events]
        # bisect_left finds first element >= claim_dt, so [:idx] contains all strictly < claim_dt
        idx_now = bisect.bisect_left(timestamps, claim_dt)
        prior_events = events[:idx_now]
        total_prior = len(prior_events)

        # 7-day window: [claim_dt - 7d, claim_dt)
        t_7d = claim_dt - timedelta(days=7)
        idx_7d = bisect.bisect_left(timestamps, t_7d, 0, idx_now)
        claims_7d = idx_now - idx_7d

        # 30-day window: [claim_dt - 30d, claim_dt)
        t_30d = claim_dt - timedelta(days=30)
        idx_30d = bisect.bisect_left(timestamps, t_30d, 0, idx_now)
        events_30d = events[idx_30d:idx_now]
        claims_30d = len(events_30d)

        sub2k_30d = sum(1 for e in events_30d if e[1])
        sub2k_ratio_30d = sub2k_30d / max(claims_30d, 1)

        # Velocity ratio: weekly rate normalized by monthly average weekly rate
        monthly_avg_weekly = (claims_30d / 4.0) + 1.0
        velocity_ratio = claims_7d / monthly_avg_weekly

        # Target metrics (strictly prior resolved training outcomes)
        target_events = self.partner_target_events.get(pid, [])
        t_target_stamps = [e[0] for e in target_events]
        idx_target = bisect.bisect_left(t_target_stamps, claim_dt)
        prior_target_events = target_events[:idx_target]

        prior_resolved = len(prior_target_events)
        prior_fraud = sum(e[1] for e in prior_target_events)

        # Empirical Bayes shrinkage towards global prior rate
        smoothed_fraud_rate = (
            prior_fraud + self.smoothing_weight * self.global_prior_rate
        ) / (prior_resolved + self.smoothing_weight)

        return {
            "partner_claims_prev_7d": claims_7d,
            "partner_claims_prev_30d": claims_30d,
            "partner_claims_lifetime_prior": total_prior,
            "partner_claim_velocity_ratio": round(velocity_ratio, 4),
            "partner_sub2000_claims_prev_30d": sub2k_30d,
            "partner_sub2000_ratio_prev_30d": round(sub2k_ratio_30d, 4),
            "partner_prior_fraud_count": prior_fraud,
            "partner_prior_resolved_claims": prior_resolved,
            "partner_smoothed_fraud_rate": round(smoothed_fraud_rate, 5),
        }

    def query_serial_prior_count(self, serial: str, claim_dt: datetime) -> int:
        """
        Count occurrences of this serial strictly before claim_dt.
        """
        stamps = self.serial_events.get(serial, [])
        idx = bisect.bisect_left(stamps, claim_dt)
        return idx


def build_feature_dict(
    row: Dict[str, Any],
    products_map: Dict[str, Dict[str, str]],
    partners_map: Dict[str, Dict[str, str]],
    history_index: TemporalHistoryIndex,
) -> Dict[str, Any]:
    """
    Extracts all production-safe features for a single claim record.
    """
    claim_dt = row["submitted_dt"]
    amount = float(row["claim_amount_inr"])
    sku = row["sku"]
    pid = row["partner_id"]
    serial = row["product_serial"]

    # Product features
    prod = products_map[sku]
    list_price = float(prod["list_price_inr"])
    warranty_months = int(prod["warranty_months"])
    days_since_purchase = float(row["days_since_purchase"])
    warranty_duration_days = warranty_months * 30.5

    claim_amount_to_list_price_ratio = round(amount / list_price, 4)
    days_to_warranty_ratio = round(days_since_purchase / warranty_duration_days, 4)
    is_near_expiry = 1 if days_to_warranty_ratio >= 0.85 else 0
    is_early_life = 1 if days_since_purchase <= 30 else 0

    # Partner features
    partner = partners_map[pid]
    onboarded_date = datetime.fromisoformat(partner["onboarded_date"]).date()
    claim_date = claim_dt.date()
    partner_age_days = (claim_date - onboarded_date).days
    is_new_180d = 1 if partner_age_days <= 180 else 0
    is_new_365d = 1 if partner_age_days <= 365 else 0

    # Temporal partner & serial history
    hist = history_index.query_partner_metrics(pid, claim_dt)
    serial_prior_cnt = history_index.query_serial_prior_count(serial, claim_dt)

    # Calendar features
    claim_month = claim_dt.month
    claim_dow = claim_dt.weekday()
    claim_hour = claim_dt.hour
    is_weekend = 1 if claim_dow in (5, 6) else 0

    # Policy features
    is_sub2k = 1 if amount < 2000.0 else 0
    threshold_gap = round(2000.0 - amount, 2) if is_sub2k else 0.0
    post_policy = 1 if claim_dt >= datetime(2026, 5, 1) else 0

    # Text sanitization
    clean_desc = clean_fault_description(row.get("claim_description", ""))

    feat = {
        # Identification (preserved for joins, not modeling input)
        "claim_id": row["claim_id"],
        # Claim basics
        "claim_amount_inr": amount,
        "log_claim_amount": round(math.log1p(amount), 4),
        "is_sub_2000": is_sub2k,
        "claim_amount_threshold_gap": threshold_gap,
        "photo_attached": 1 if row.get("photo_attached") == "Y" else 0,
        "customer_prior_claims": int(row.get("customer_prior_claims", 0)),
        # Product & Warranty
        "sku": sku,
        "product_family": prod["family"],
        "product_list_price": list_price,
        "product_warranty_months": warranty_months,
        "claim_amount_to_list_price_ratio": claim_amount_to_list_price_ratio,
        "days_since_purchase": int(days_since_purchase),
        "days_to_warranty_ratio": days_to_warranty_ratio,
        "is_near_warranty_expiry": is_near_expiry,
        "is_early_life_claim": is_early_life,
        # Partner Profile
        "partner_type": partner["partner_type"],
        "partner_city": partner["city"],
        "partner_age_days_at_claim": partner_age_days,
        "is_new_partner_180d": is_new_180d,
        "is_new_partner_365d": is_new_365d,
        # Partner History & Velocity
        "partner_claims_prev_7d": hist["partner_claims_prev_7d"],
        "partner_claims_prev_30d": hist["partner_claims_prev_30d"],
        "partner_claims_lifetime_prior": hist["partner_claims_lifetime_prior"],
        "partner_claim_velocity_ratio": hist["partner_claim_velocity_ratio"],
        "partner_sub2000_claims_prev_30d": hist["partner_sub2000_claims_prev_30d"],
        "partner_sub2000_ratio_prev_30d": hist["partner_sub2000_ratio_prev_30d"],
        # Target-Based Partner Features
        "partner_prior_fraud_count": hist["partner_prior_fraud_count"],
        "partner_prior_resolved_claims": hist["partner_prior_resolved_claims"],
        "partner_smoothed_fraud_rate": hist["partner_smoothed_fraud_rate"],
        # Serial History
        "serial_prior_claim_count": serial_prior_cnt,
        # Calendar & Policy
        "claim_month": claim_month,
        "claim_day_of_week": claim_dow,
        "claim_hour": claim_hour,
        "is_weekend": is_weekend,
        "post_policy_change": post_policy,
        # Text
        "clean_fault_description": clean_desc,
    }

    return feat


def generate_feature_manifest() -> List[Dict[str, str]]:
    """
    Creates comprehensive manifest detailing every feature's metadata, formula,
    availability timestamp, leakage risk, and production approval status.
    """
    manifest = [
        {
            "feature_name": "claim_amount_inr",
            "data_source": "train.csv / test_unlabelled.csv",
            "formula": "claim_amount_inr",
            "meaning": "Declared warranty repair claim amount in Indian Rupees",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None (primary production input)",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "log_claim_amount",
            "data_source": "Derived from claim_amount_inr",
            "formula": "log1p(claim_amount_inr)",
            "meaning": "Natural logarithm of claim amount for variance stabilization",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "is_sub_2000",
            "data_source": "Derived from claim_amount_inr",
            "formula": "1 if claim_amount_inr < 2000.0 else 0",
            "meaning": "Indicator whether claim falls below May 1, 2026 auto-approval threshold",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "claim_amount_threshold_gap",
            "data_source": "Derived from claim_amount_inr",
            "formula": "max(0.0, 2000.0 - claim_amount_inr) if is_sub_2000 else 0.0",
            "meaning": "Distance below ₹2,000 threshold (sensitive to threshold clustering)",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "photo_attached",
            "data_source": "train.csv / test_unlabelled.csv",
            "formula": "1 if photo_attached == 'Y' else 0",
            "meaning": "Binary flag indicating whether diagnostic photograph was submitted",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "customer_prior_claims",
            "data_source": "train.csv / test_unlabelled.csv",
            "formula": "customer_prior_claims",
            "meaning": "Historical claim count for customer in Kestrel CRM",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "sku",
            "data_source": "train.csv / test_unlabelled.csv",
            "formula": "sku",
            "meaning": "Product Stock Keeping Unit identifier (21 distinct values)",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "product_family",
            "data_source": "products.csv",
            "formula": "lookup(sku, products.family)",
            "meaning": "Appliance family (7 categories: Air Fryer, Water Purifier, etc.)",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "product_list_price",
            "data_source": "products.csv",
            "formula": "lookup(sku, products.list_price_inr)",
            "meaning": "Catalog retail price of the appliance in Indian Rupees",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "product_warranty_months",
            "data_source": "products.csv",
            "formula": "lookup(sku, products.warranty_months)",
            "meaning": "Official manufacturer warranty duration (12 or 24 months)",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "claim_amount_to_list_price_ratio",
            "data_source": "Derived from claim and products.csv",
            "formula": "claim_amount_inr / product_list_price",
            "meaning": "Claim cost relative to appliance replacement cost",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "days_since_purchase",
            "data_source": "train.csv / test_unlabelled.csv",
            "formula": "days_since_purchase",
            "meaning": "Days elapsed between customer invoice date and claim",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "days_to_warranty_ratio",
            "data_source": "Derived from claim and products.csv",
            "formula": "days_since_purchase / (product_warranty_months * 30.5)",
            "meaning": "Fraction of warranty period expired at time of failure",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "is_near_warranty_expiry",
            "data_source": "Derived from days_to_warranty_ratio",
            "formula": "1 if days_to_warranty_ratio >= 0.85 else 0",
            "meaning": "Claim filed in final 15% of warranty window",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "is_early_life_claim",
            "data_source": "Derived from days_since_purchase",
            "formula": "1 if days_since_purchase <= 30 else 0",
            "meaning": "Infant mortality claim filed within 30 days of purchase",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "partner_type",
            "data_source": "partners.csv",
            "formula": "lookup(partner_id, partners.partner_type)",
            "meaning": "Partner category: authorised_service_centre, franchise, freelance_technician",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "partner_city",
            "data_source": "partners.csv",
            "formula": "lookup(partner_id, partners.city)",
            "meaning": "Operating city of the partner (18 cities across India)",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "partner_age_days_at_claim",
            "data_source": "Derived from claim and partners.csv",
            "formula": "(submitted_at.date - onboarded_date).days",
            "meaning": "Partner operational tenure in days on the day claim is filed",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "is_new_partner_180d",
            "data_source": "Derived from partner_age_days_at_claim",
            "formula": "1 if partner_age_days_at_claim <= 180 else 0",
            "meaning": "Flag indicating partner onboarded within past 6 months",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "is_new_partner_365d",
            "data_source": "Derived from partner_age_days_at_claim",
            "formula": "1 if partner_age_days_at_claim <= 365 else 0",
            "meaning": "Flag indicating partner onboarded within past 1 year",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "partner_claims_prev_7d",
            "data_source": "Temporal claim intake index",
            "formula": "count(claims by partner in [T - 7d, T))",
            "meaning": "Short-term burst volume: partner claims submitted in preceding 7 days",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None (strictly prior claims, current excluded)",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "partner_claims_prev_30d",
            "data_source": "Temporal claim intake index",
            "formula": "count(claims by partner in [T - 30d, T))",
            "meaning": "Medium-term volume: partner claims submitted in preceding 30 days",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None (strictly prior claims, current excluded)",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "partner_claims_lifetime_prior",
            "data_source": "Temporal claim intake index",
            "formula": "count(claims by partner strictly before T)",
            "meaning": "Cumulative historical experience of partner prior to current claim",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None (strictly prior claims, current excluded)",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "partner_claim_velocity_ratio",
            "data_source": "Derived from 7d and 30d counts",
            "formula": "claims_7d / (claims_30d / 4.0 + 1.0)",
            "meaning": "Ratio of recent 7-day intake rate to expected 30-day weekly baseline",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None (strictly prior claims)",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "partner_sub2000_claims_prev_30d",
            "data_source": "Temporal claim intake index",
            "formula": "count(claims < 2000 by partner in [T - 30d, T))",
            "meaning": "Small claim volume submitted by partner in preceding 30 days",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None (strictly prior claims)",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "partner_sub2000_ratio_prev_30d",
            "data_source": "Derived from sub2000 and total 30d counts",
            "formula": "sub2k_claims_30d / max(claims_30d, 1)",
            "meaning": "Share of recent partner claims exploiting auto-approval threshold",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None (strictly prior claims)",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "partner_prior_fraud_count",
            "data_source": "Historical resolved investigation log",
            "formula": "count(claims by partner before T with is_fraud == 1)",
            "meaning": "Historical confirmed fraud occurrences for this partner strictly before T",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None (strictly prior resolved training outcomes)",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "partner_prior_resolved_claims",
            "data_source": "Historical resolved investigation log",
            "formula": "count(claims by partner before T with is_fraud in (0, 1))",
            "meaning": "Historical audit sample size for this partner strictly before T",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None (strictly prior resolved training outcomes)",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "partner_smoothed_fraud_rate",
            "data_source": "Derived via Empirical Bayes shrinkage",
            "formula": "(prior_fraud + 20 * 0.01265) / (prior_resolved + 20)",
            "meaning": "Bayesian-smoothed partner fraud rate with shrinkage towards global mean",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None (strictly prior resolved training outcomes)",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "serial_prior_claim_count",
            "data_source": "Temporal serial intake index",
            "formula": "count(claims with same product_serial strictly before T)",
            "meaning": "Frequency of repeat claims filed under the same serial number",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None (strictly prior claims)",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "claim_month",
            "data_source": "Derived from submitted_at",
            "formula": "submitted_at.month",
            "meaning": "Calendar month of claim submission (1 to 12)",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "claim_day_of_week",
            "data_source": "Derived from submitted_at",
            "formula": "submitted_at.weekday",
            "meaning": "Day of week of claim submission (0=Monday, 6=Sunday)",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "claim_hour",
            "data_source": "Derived from submitted_at",
            "formula": "submitted_at.hour",
            "meaning": "Hour of day when claim was lodged in Indian Standard Time (0 to 23)",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "is_weekend",
            "data_source": "Derived from claim_day_of_week",
            "formula": "1 if claim_day_of_week in (5, 6) else 0",
            "meaning": "Flag indicating weekend submission",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "post_policy_change",
            "data_source": "Derived from submitted_at",
            "formula": "1 if submitted_at >= '2026-05-01' else 0",
            "meaning": "Flag indicating claim was submitted under the post-May 1, 2026 policy",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None",
            "allowed_in_production": "Yes",
        },
        {
            "feature_name": "clean_fault_description",
            "data_source": "Derived from claim_description",
            "formula": "clean_fault_description(claim_description)",
            "meaning": "Standardized canonical appliance failure category (12 categories)",
            "availability_timestamp": "At claim submission",
            "leakage_risk": "None (sanitized, prompt injections stripped)",
            "allowed_in_production": "Yes",
        },
        # EXCLUDED / REJECTED FEATURES
        {
            "feature_name": "inspector_note",
            "data_source": "train.csv / test_unlabelled.csv",
            "formula": "inspector_note (raw text)",
            "meaning": "Technician/inspector manual field notes entered after inspection",
            "availability_timestamp": "After inspection sign-off",
            "leakage_risk": "SEVERE (81.2% missing in test due to auto-approval rule)",
            "allowed_in_production": "NO (Rejected)",
        },
        {
            "feature_name": "partner_inspected",
            "data_source": "train.csv / test_unlabelled.csv",
            "formula": "partner_inspected (raw 'Y'/'N')",
            "meaning": "Inspection sign-off flag",
            "availability_timestamp": "After inspection sign-off",
            "leakage_risk": "SEVERE (Distribution shift: 83.7% in train -> 21.3% in test)",
            "allowed_in_production": "NO (Rejected)",
        },
        {
            "feature_name": "source",
            "data_source": "train.csv",
            "formula": "source ('crm' vs 'legacy_zoho')",
            "meaning": "IT database migration origin flag",
            "availability_timestamp": "Data migration artifact",
            "leakage_risk": "HIGH (100% CRM in test, 0% legacy_zoho; spurious artifact)",
            "allowed_in_production": "NO (Rejected)",
        },
        {
            "feature_name": "is_fraud",
            "data_source": "train.csv",
            "formula": "is_fraud",
            "meaning": "Confirmed ground truth fraud status",
            "availability_timestamp": "Only after forensic fraud investigation",
            "leakage_risk": "DIRECT TARGET LEAKAGE (Supervised target label only)",
            "allowed_in_production": "NO (Target)",
        },
        {
            "feature_name": "lifetime_partner_fraud_rate",
            "data_source": "Calculated across entire training set",
            "formula": "sum(fraud) / count(claims)",
            "meaning": "Partner lifetime fraud rate using future data",
            "availability_timestamp": "At end of study",
            "leakage_risk": "SEVERE TEMPORAL LEAKAGE (Leaks future outcomes into past)",
            "allowed_in_production": "NO (Rejected)",
        },
        {
            "feature_name": "duplicate_resubmission_records",
            "data_source": "train.csv",
            "formula": "Second submission of same claim_id",
            "meaning": "Resubmitted duplicate claims",
            "availability_timestamp": "3 days post initial submission",
            "leakage_risk": "Artificial weight distortion and train/val cross-contamination",
            "allowed_in_production": "NO (Excluded from canonical train)",
        },
    ]
    return manifest


def execute_feature_pipeline():
    print("=" * 70)
    print("EXECUTING PHASE 2A LEAKAGE-SAFE FEATURE PIPELINE")
    print("=" * 70)

    # 1. Load source datasets
    _, train_raw = load_csv("train.csv")
    _, test_raw = load_csv("test_unlabelled.csv")
    _, partners_raw = load_csv("partners.csv")
    _, products_raw = load_csv("products.csv")

    partners_map = {p["partner_id"]: p for p in partners_raw}
    products_map = {p["sku"]: p for p in products_raw}

    print(f"Loaded {len(train_raw)} raw train rows and {len(test_raw)} test rows.")

    # 2. Parse timestamps and prepare records
    for r in train_raw:
        r["submitted_dt"] = datetime.fromisoformat(r["submitted_at"].replace("Z", ""))
    for r in test_raw:
        r["submitted_dt"] = datetime.fromisoformat(r["submitted_at"].replace("Z", ""))

    # 3. Deduplicate train claims: Canonical selection (earliest submission)
    train_by_cid = collections.defaultdict(list)
    for r in train_raw:
        train_by_cid[r["claim_id"]].append(r)

    canonical_train = []
    excluded_duplicate_rows = []
    canonical_metadata = []

    for cid, rows in train_by_cid.items():
        rows_sorted = sorted(rows, key=lambda x: x["submitted_dt"])
        canonical = rows_sorted[0]
        canonical_train.append(canonical)

        has_resub = len(rows_sorted) > 1
        gap_hours = (
            (rows_sorted[1]["submitted_dt"] - rows_sorted[0]["submitted_dt"]).total_seconds() / 3600.0
            if has_resub
            else 0.0
        )

        canonical_metadata.append(
            {
                "claim_id": cid,
                "first_submitted_at": canonical["submitted_at"],
                "source": canonical["source"],
                "has_resubmission": 1 if has_resub else 0,
                "resubmission_count": len(rows_sorted),
                "resubmission_delay_hours": round(gap_hours, 2),
                "is_fraud": canonical["is_fraud"],
                "split_assignment": (
                    "train_split"
                    if canonical["submitted_dt"] < datetime(2026, 5, 1)
                    else "validation_split"
                ),
            }
        )

        if has_resub:
            for extra in rows_sorted[1:]:
                excluded_duplicate_rows.append(
                    {
                        "claim_id": extra["claim_id"],
                        "submitted_at": extra["submitted_at"],
                        "partner_id": extra["partner_id"],
                        "sku": extra["sku"],
                        "claim_amount_inr": extra["claim_amount_inr"],
                        "exclusion_reason": "duplicate_resubmission_second_entry",
                        "is_fraud": extra["is_fraud"],
                    }
                )

    print(f"Canonical train claims: {len(canonical_train)} unique claims.")
    print(f"Excluded duplicate second submissions: {len(excluded_duplicate_rows)}.")

    # 4. Handle target labels & isolate undecided records
    labeled_canonical_train = [r for r in canonical_train if r["is_fraud"] in ("0", "1")]
    undecided_canonical_train = [r for r in canonical_train if r["is_fraud"] == ""]

    print(f"Labeled canonical train claims: {len(labeled_canonical_train)}")
    print(f"Undecided canonical train claims: {len(undecided_canonical_train)}")

    # Add undecided claims to excluded_records
    excluded_records = list(excluded_duplicate_rows)
    for u in undecided_canonical_train:
        excluded_records.append(
            {
                "claim_id": u["claim_id"],
                "submitted_at": u["submitted_at"],
                "partner_id": u["partner_id"],
                "sku": u["sku"],
                "claim_amount_inr": u["claim_amount_inr"],
                "exclusion_reason": "open_case_undecided_target",
                "is_fraud": "",
            }
        )

    # 5. Build Temporal History Index
    # All claims (canonical train + test) are indexed chronologically.
    # Training label cutoff is 2026-07-01: no test claim ever updates fraud outcomes!
    print("Building temporal history index...")
    history_index = TemporalHistoryIndex(
        global_prior_rate=0.01265, smoothing_weight=20.0
    )
    all_intake_records = canonical_train + test_raw
    history_index.populate(all_intake_records, label_cutoff=datetime(2026, 7, 1))

    # 6. Extract production-safe features
    print("Extracting features for labeled training set...")
    train_features = []
    for r in labeled_canonical_train:
        feat = build_feature_dict(r, products_map, partners_map, history_index)
        # Include target integer label in train_features.csv for supervised modeling
        feat["is_fraud"] = int(r["is_fraud"])
        train_features.append(feat)

    print("Extracting features for test set...")
    test_features = []
    for r in test_raw:
        feat = build_feature_dict(r, products_map, partners_map, history_index)
        test_features.append(feat)

    # 7. Write Output Datasets
    os.makedirs(FEATURES_OUTPUT_DIR, exist_ok=True)

    # 7.1 train_features.csv
    train_feat_path = os.path.join(FEATURES_OUTPUT_DIR, "train_features.csv")
    train_feat_cols = ["claim_id", "is_fraud"] + PRODUCTION_FEATURE_COLUMNS
    with open(train_feat_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=train_feat_cols)
        writer.writeheader()
        writer.writerows(train_features)
    print(f"Written: {train_feat_path} ({len(train_features)} rows, {len(train_feat_cols)} columns)")

    # 7.2 test_features.csv
    test_feat_path = os.path.join(FEATURES_OUTPUT_DIR, "test_features.csv")
    test_feat_cols = ["claim_id"] + PRODUCTION_FEATURE_COLUMNS
    with open(test_feat_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=test_feat_cols)
        writer.writeheader()
        writer.writerows(test_features)
    print(f"Written: {test_feat_path} ({len(test_features)} rows, {len(test_feat_cols)} columns)")

    # 7.3 feature_manifest.csv
    manifest = generate_feature_manifest()
    manifest_path = os.path.join(FEATURES_OUTPUT_DIR, "feature_manifest.csv")
    manifest_cols = [
        "feature_name",
        "data_source",
        "formula",
        "meaning",
        "availability_timestamp",
        "leakage_risk",
        "allowed_in_production",
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_cols)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Written: {manifest_path} ({len(manifest)} rows)")

    # 7.4 train_modeling_metadata.csv
    meta_path = os.path.join(FEATURES_OUTPUT_DIR, "train_modeling_metadata.csv")
    meta_cols = [
        "claim_id",
        "first_submitted_at",
        "source",
        "has_resubmission",
        "resubmission_count",
        "resubmission_delay_hours",
        "is_fraud",
        "split_assignment",
    ]
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=meta_cols)
        writer.writeheader()
        writer.writerows(canonical_metadata)
    print(f"Written: {meta_path} ({len(canonical_metadata)} rows)")

    # 7.5 excluded_records.csv
    excluded_path = os.path.join(FEATURES_OUTPUT_DIR, "excluded_records.csv")
    excluded_cols = [
        "claim_id",
        "submitted_at",
        "partner_id",
        "sku",
        "claim_amount_inr",
        "exclusion_reason",
        "is_fraud",
    ]
    with open(excluded_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=excluded_cols)
        writer.writeheader()
        writer.writerows(excluded_records)
    print(f"Written: {excluded_path} ({len(excluded_records)} rows)")

    # 7.6 partner_history_diagnostics.csv
    partner_diag = []
    # Calculate partner summary as of 2026-06-30
    cutoff = datetime(2026, 7, 1)
    test_partners = {r["partner_id"] for r in test_raw}
    for pid, p in partners_map.items():
        hist = history_index.query_partner_metrics(pid, cutoff)
        is_cold = 1 if pid in test_partners and hist["partner_claims_lifetime_prior"] == 0 else 0
        partner_diag.append(
            {
                "partner_id": pid,
                "city": p["city"],
                "partner_type": p["partner_type"],
                "onboarded_date": p["onboarded_date"],
                "total_train_claims": hist["partner_claims_lifetime_prior"],
                "prior_resolved_claims": hist["partner_prior_resolved_claims"],
                "prior_fraud_claims": hist["partner_prior_fraud_count"],
                "smoothed_fraud_rate_at_train_end": hist["partner_smoothed_fraud_rate"],
                "is_cold_start_in_test": is_cold,
            }
        )

    diag_path = os.path.join(FEATURES_OUTPUT_DIR, "partner_history_diagnostics.csv")
    diag_cols = [
        "partner_id",
        "city",
        "partner_type",
        "onboarded_date",
        "total_train_claims",
        "prior_resolved_claims",
        "prior_fraud_claims",
        "smoothed_fraud_rate_at_train_end",
        "is_cold_start_in_test",
    ]
    with open(diag_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=diag_cols)
        writer.writeheader()
        writer.writerows(partner_diag)
    print(f"Written: {diag_path} ({len(partner_diag)} rows)")

    print("=" * 70)
    print("PHASE 2A FEATURE PIPELINE EXECUTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    execute_feature_pipeline()
