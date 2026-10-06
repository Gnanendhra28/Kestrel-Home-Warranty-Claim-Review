"""
Business Logic & Fraud Risk Explanation Engine for Kestrel Home Warranty Fraud.

Implements:
- Expected value screening: expected_fraud_value = fraud_risk_score * claim_amount_inr
- Review recommendation based on desk review cost (₹380)
- Risk tiering (HIGH, MEDIUM, LOWER)
- Deterministic, human-readable reason generation (3-5 reasons, strictly no 'confirmed fraud' claims)
- Operational governance disclaimers
"""

from typing import Dict, List, Any, Tuple

REVIEW_COST_INR = 380.0

DISCLAIMER_TEXT = (
    "DISCLAIMER: This assessment provides statistical risk-prioritization and decision-support "
    "guidance for internal triage only. A high score or review recommendation reflects elevated "
    "risk indicators and DOES NOT constitute confirmed fraud. Formal claim rejection or payout "
    "denial requires thorough physical investigation per Kestrel Warranty Operations Policy."
)


def compute_expected_fraud_value(fraud_risk_score: float, claim_amount_inr: float) -> float:
    """
    Computes the estimated expected financial fraud loss for a claim in INR:
    E[Fraud Value] = P(Fraud | X) * Claim Amount
    """
    return round(float(fraud_risk_score) * float(claim_amount_inr), 2)


def determine_risk_level(fraud_risk_score: float) -> str:
    """
    Categorizes continuous fraud risk probability into operational risk tiers.
    """
    if fraud_risk_score >= 0.40:
        return "HIGH"
    elif fraud_risk_score >= 0.20:
        return "MEDIUM"
    else:
        return "LOWER"


def determine_review_recommendation(expected_fraud_value: float, review_cost: float = REVIEW_COST_INR) -> bool:
    """
    Screens whether manual desk review is economically justified:
    Recommended if E[Fraud Value] > ₹380 manual review cost.
    """
    return expected_fraud_value > review_cost


def generate_risk_reasons(
    feat_dict: Dict[str, Any],
    fraud_risk_score: float,
    expected_fraud_value: float,
    min_reasons: int = 3,
    max_reasons: int = 5,
) -> List[str]:
    """
    Generates deterministic, factual, evidence-based explanations for a claim's score.
    Returns 3 to 5 ranked reasons based on domain risk severity.
    Guarantees no false assertions of 'confirmed fraud'.
    """
    candidates: List[Tuple[int, str]] = []

    amount = float(feat_dict.get("claim_amount_inr", 0.0))
    list_price = float(feat_dict.get("product_list_price", 0.0))
    price_ratio = float(feat_dict.get("claim_amount_to_list_price_ratio", 0.0))
    prior_serial = int(feat_dict.get("serial_prior_claim_count", 0))
    partner_prior_fraud = int(feat_dict.get("partner_prior_fraud_count", 0))
    partner_smoothed_rate = float(feat_dict.get("partner_smoothed_fraud_rate", 0.0))
    claims_7d = int(feat_dict.get("partner_claims_prev_7d", 0))
    velocity_ratio = float(feat_dict.get("partner_claim_velocity_ratio", 1.0))
    photo_attached = int(feat_dict.get("photo_attached", 1))
    cust_prior_claims = int(feat_dict.get("customer_prior_claims", 0))
    is_sub2k = int(feat_dict.get("is_sub_2000", 0))
    threshold_gap = float(feat_dict.get("claim_amount_threshold_gap", 0.0))
    days_since_purchase = int(feat_dict.get("days_since_purchase", 0))
    is_near_expiry = int(feat_dict.get("is_near_warranty_expiry", 0))
    is_early_life = int(feat_dict.get("is_early_life_claim", 0))
    warranty_months = int(feat_dict.get("product_warranty_months", 12))

    # 1. Product serial recycling
    if prior_serial >= 1:
        candidates.append((
            95,
            f"Product serial has appeared in {prior_serial} prior warranty claim(s), suggesting potential duplicate/recycled serial filing."
        ))

    # 2. Elevated partner historical fraud rate
    if partner_prior_fraud > 0 or partner_smoothed_rate >= 0.025:
        candidates.append((
            90,
            f"Service partner has {partner_prior_fraud} prior confirmed fraud incident(s) with an empirical Bayes smoothed risk rate of {partner_smoothed_rate:.1%}."
        ))

    # 3. High claim amount relative to product list price
    if list_price > 0 and price_ratio >= 0.60:
        candidates.append((
            85,
            f"Claim amount (₹{amount:,.0f}) is {price_ratio:.1%} of product list price (₹{list_price:,.0f}), well above normal repair component ratios."
        ))
    elif list_price > 0 and price_ratio >= 0.40:
        candidates.append((
            65,
            f"Claim amount represents {price_ratio:.1%} of product list price (₹{amount:,.0f} vs ₹{list_price:,.0f})."
        ))

    # 4. Sub-₹2,000 threshold bunching
    if is_sub2k and amount >= 1700.0:
        candidates.append((
            80,
            f"Claim amount (₹{amount:,.0f}) clusters just below the ₹2,000 fast-track auto-approval threshold (gap: ₹{threshold_gap:,.0f})."
        ))

    # 5. Missing photo verification
    if photo_attached == 0:
        candidates.append((
            75,
            "Claim submitted without supporting photographic verification, increasing documentation risk."
        ))

    # 6. Partner volume burst / surge
    if velocity_ratio >= 2.0 or (claims_7d >= 10):
        candidates.append((
            70,
            f"Partner claim velocity surge detected: {claims_7d} claims submitted in the prior 7 days ({velocity_ratio:.1f}x baseline rate)."
        ))
    elif velocity_ratio >= 1.5:
        candidates.append((
            60,
            f"Partner shows an elevated 7-day submission velocity ({velocity_ratio:.1f}x baseline rate with {claims_7d} claims)."
        ))

    # 7. Customer claim frequency
    if cust_prior_claims >= 3:
        candidates.append((
            68,
            f"Customer profile indicates high claim frequency ({cust_prior_claims} prior warranty claims)."
        ))
    elif cust_prior_claims >= 2:
        candidates.append((
            50,
            f"Customer has submitted {cust_prior_claims} previous warranty claims."
        ))

    # 8. High absolute value exposure
    if amount >= 4500.0:
        candidates.append((
            62,
            f"High monetary value (₹{amount:,.0f}) creates substantial financial loss exposure if erroneous."
        ))

    # 9. Warranty timing anomalies
    if is_near_expiry:
        candidates.append((
            55,
            f"Claim submitted in the final phase of warranty coverage ({days_since_purchase} days post-purchase on a {warranty_months}-month policy)."
        ))
    elif is_early_life:
        candidates.append((
            45,
            f"Early-life claim submitted within 30 days of purchase ({days_since_purchase} days)."
        ))

    # 10. Economic review threshold reasoning
    if expected_fraud_value > REVIEW_COST_INR:
        candidates.append((
            40,
            f"Estimated expected fraud value of ₹{expected_fraud_value:,.0f} exceeds the ₹{REVIEW_COST_INR:,.0f} investigation desk review cost."
        ))
    else:
        candidates.append((
            30,
            f"Estimated expected fraud value of ₹{expected_fraud_value:,.0f} is below the ₹{REVIEW_COST_INR:,.0f} manual desk review threshold."
        ))

    # General baseline if candidates are sparse
    if len(candidates) < min_reasons:
        if fraud_risk_score < 0.20:
            candidates.append((
                20,
                f"Statistical risk score ({fraud_risk_score:.2f}) falls within baseline operating bounds across historical service partners."
            ))
            candidates.append((
                10,
                "Claim parameters (documentation, pricing, and timing) align with routine warranty repair profiles."
            ))
        else:
            candidates.append((
                20,
                f"Multi-feature interaction patterns generate an aggregate statistical risk index of {fraud_risk_score:.2f}."
            ))

    # Sort descending by priority/severity
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Return top 3-5 unique reasons
    selected = []
    seen = set()
    for _, reason in candidates:
        if reason not in seen:
            seen.add(reason)
            selected.append(reason)
        if len(selected) >= max_reasons:
            break

    return selected[:max(min_reasons, min(len(selected), max_reasons))]
