"""
Kestrel Home Warranty — Fraud Risk Review Dashboard (Phase 3).

Employee-facing decision support screen for warranty claim triage prior to payout.
Calls the FastAPI backend (http://127.0.0.1:8000/predict) with transparent
direct-engine fallback if the API server is offline.
"""

from datetime import datetime
import json
import os
import requests
import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Fraud Risk Review — Kestrel Home",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.environ.get("KESTREL_API_URL", "http://127.0.0.1:8000")

# Custom styling for high clarity
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #0d6efd;
        margin-bottom: 12px;
    }
    .status-badge-high {
        background-color: #f8d7da;
        color: #842029;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 700;
        display: inline-block;
    }
    .status-badge-med {
        background-color: #fff3cd;
        color: #664d03;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 700;
        display: inline-block;
    }
    .status-badge-low {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 700;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #f1f3f5;
        border-left: 4px solid #6c757d;
        padding: 12px;
        font-size: 0.85rem;
        color: #495057;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.5)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}, "HTTP"
    except Exception:
        try:
            from fastapi.testclient import TestClient
            from src.api.main import app as fastapi_app
            with TestClient(fastapi_app) as client:
                res = client.get("/health")
                return res.status_code == 200, res.json() if res.status_code == 200 else {}, "In-Process"
        except Exception:
            return False, {}, "Offline"


def call_predict_api(payload):
    # Try HTTP first
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=2.0)
        if r.status_code == 200:
            return r.json(), None
        else:
            return None, f"API Error ({r.status_code}): {r.text}"
    except Exception:
        # Fallback to direct in-process FastAPI TestClient for offline/sandboxed environments
        try:
            from fastapi.testclient import TestClient
            from src.api.main import app as fastapi_app
            with TestClient(fastapi_app) as client:
                res = client.post("/predict", json=payload)
                if res.status_code == 200:
                    return res.json(), None
                else:
                    return None, f"In-Process Evaluation Error ({res.status_code}): {res.text}"
        except Exception as fallback_err:
            return None, f"Evaluation failed: {fallback_err}"


# Pre-configured examples
SAMPLE_PRESETS = {
    "Select a preset or custom input...": None,
    "High Risk — Recycled Serial & High Value": {
        "claim_id": "WC-DEMO-01",
        "submitted_at": "2026-07-20T14:30:00",
        "partner_id": "SP3033",
        "sku": "KH-AF-01",
        "product_serial": "AF-2023-8812",
        "days_since_purchase": 350,
        "claim_amount_inr": 4800.0,
        "photo_attached": "N",
        "customer_prior_claims": 3,
        "claim_description": "unit not heating tripping mcb",
    },
    "Medium Risk — Sub-₹2,000 Threshold Bunching": {
        "claim_id": "WC-DEMO-02",
        "submitted_at": "2026-07-18T10:15:00",
        "partner_id": "SP3032",
        "sku": "KH-IC-01",
        "product_serial": "IC-2024-4419",
        "days_since_purchase": 180,
        "claim_amount_inr": 1950.0,
        "photo_attached": "N",
        "customer_prior_claims": 1,
        "claim_description": "power button not working",
    },
    "Lower Risk — Routine Low-Value Repair with Photo": {
        "claim_id": "WC-DEMO-03",
        "submitted_at": "2026-07-16T16:00:00",
        "partner_id": "SP3265",
        "sku": "KH-CF-01",
        "product_serial": "CF-2024-0012",
        "days_since_purchase": 90,
        "claim_amount_inr": 850.0,
        "photo_attached": "Y",
        "customer_prior_claims": 0,
        "claim_description": "blade jammed loud noise while running",
    },
}

VALID_SKUS = [
    "KH-AF-01", "KH-AF-02", "KH-AF-03",
    "KH-MG-01", "KH-MG-02", "KH-MG-03",
    "KH-WP-01", "KH-WP-02", "KH-WP-03",
    "KH-RV-01", "KH-RV-02", "KH-RV-03",
    "KH-IC-01", "KH-IC-02", "KH-IC-03",
    "KH-CF-01", "KH-CF-02", "KH-CF-03",
    "KH-RH-01", "KH-RH-02", "KH-RH-03",
]

# Header
st.title("Fraud Risk Review")
st.markdown(
    "**Kestrel Home Appliances — Pre-Payout Warranty Claim Review & Decision Support**"
)

# Operational status banner
api_ok, health_data, conn_mode = check_api_health()
col_stat1, col_stat2, col_stat3 = st.columns([2, 1, 1])
with col_stat1:
    if api_ok:
        st.success(f"Backend Engine Connected ({conn_mode}) — Model Pipeline Active (33 features)")
    else:
        st.warning(f"Backend API Offline ({API_URL}) — Please ensure FastAPI server is running on port 8000.")
with col_stat2:
    st.info("Desk Capacity: **40 claims/month**")
with col_stat3:
    st.info("Review Cost: **₹380 / claim**")

st.divider()

# Sidebar: Claim Input Form
st.sidebar.header("Claim Submission Details")

preset_choice = st.sidebar.selectbox("Load Example Claim", list(SAMPLE_PRESETS.keys()))
default_vals = SAMPLE_PRESETS.get(preset_choice) or {
    "claim_id": "WC-MANUAL-001",
    "submitted_at": "2026-07-25T11:00:00",
    "partner_id": "SP3033",
    "sku": "KH-AF-01",
    "product_serial": "AF-2024-5501",
    "days_since_purchase": 120,
    "claim_amount_inr": 3600.0,
    "photo_attached": "Y",
    "customer_prior_claims": 0,
    "claim_description": "unit not heating",
}

with st.sidebar.form("claim_form"):
    f_claim_id = st.text_input("Claim ID", value=default_vals["claim_id"])
    f_submitted_at = st.text_input("Submission Timestamp (ISO)", value=default_vals["submitted_at"])
    f_partner_id = st.text_input("Partner ID", value=default_vals["partner_id"])
    sku_idx = VALID_SKUS.index(default_vals["sku"]) if default_vals["sku"] in VALID_SKUS else 0
    f_sku = st.selectbox("Product SKU", VALID_SKUS, index=sku_idx)
    f_product_serial = st.text_input("Product Serial Number", value=default_vals["product_serial"])
    f_amount = st.number_input("Claim Amount (₹ INR)", min_value=1.0, value=float(default_vals["claim_amount_inr"]), step=100.0)
    f_days_since_purchase = st.number_input("Days Since Purchase", min_value=0, value=int(default_vals["days_since_purchase"]))
    f_photo = st.selectbox("Photo Attached?", ["Y", "N"], index=0 if default_vals["photo_attached"] == "Y" else 1)
    f_customer_claims = st.number_input("Customer Prior Claims", min_value=0, value=int(default_vals["customer_prior_claims"]))
    f_desc = st.text_input("Fault Description", value=default_vals["claim_description"])

    submit_button = st.form_submit_button("Assess Claim Risk", type="primary", use_container_width=True)

# Main Body: Evaluation Output
payload = {
    "claim_id": f_claim_id,
    "submitted_at": f_submitted_at,
    "partner_id": f_partner_id.strip(),
    "sku": f_sku,
    "product_serial": f_product_serial.strip(),
    "days_since_purchase": float(f_days_since_purchase),
    "claim_amount_inr": float(f_amount),
    "photo_attached": f_photo,
    "customer_prior_claims": int(f_customer_claims),
    "claim_description": f_desc,
}

# Auto-evaluate on submit or default load
if submit_button or preset_choice != "Select a preset or custom input...":
    with st.spinner("Evaluating claim against production fraud model..."):
        res, err = call_predict_api(payload)

    if err:
        st.error(f"Assessment Failed: {err}")
    elif res:
        score = res["fraud_risk_score"]
        risk_lvl = res["risk_level"]
        rec = res["review_recommended"]
        exp_val = res["estimated_expected_fraud_value_inr"]
        review_cost = res["review_cost_inr"]
        net_exp_val = exp_val - review_cost

        # Row 1: High Level Verdict Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Fraud Risk Score", f"{score:.4f}", help="Calibrated posterior probability from verified Random Forest ensemble")
        with c2:
            st.metric("Assigned Risk Tier", risk_lvl)
        with c3:
            st.metric("Est. Expected Fraud Loss", f"₹{exp_val:,.2f}", help="Score × Claim Amount")
        with c4:
            st.metric("Desk Review Cost", f"₹{review_cost:,.0f}", delta=f"Net: ₹{net_exp_val:,.2f}", delta_color="normal" if net_exp_val > 0 else "off")

        # Recommendation Banner
        st.subheader("Operational Recommendation")
        if rec:
            st.error(
                f"🚨 **RECOMMEND MANUAL DESK REVIEW** — Expected fraud loss (₹{exp_val:,.2f}) exceeds the review cost (₹{review_cost:,.0f}). "
                f"Queue this claim for senior investigator inspection before approving payout."
            )
        else:
            st.success(
                f"✅ **FAST-TRACK / STANDARD PAYOUT** — Expected fraud loss (₹{exp_val:,.2f}) does not justify the ₹{review_cost:,.0f} manual desk review fee. "
                f"Low priority for manual review."
            )

        # Row 2: Reasons
        st.subheader("Key Fraud Risk Reasons (Evidence-Based)")
        st.write("The decision engine identified the following top operational indicators for this score:")
        for idx, reason in enumerate(res["reasons"], 1):
            st.markdown(f"**{idx}.** {reason}")

        # Row 3: Economics breakdown
        with st.expander("Economic & Capacity Analysis for Desk Reviewers", expanded=True):
            st.markdown(f"""
            - **Claim Amount:** ₹{res['claim_amount_inr']:,.2f}
            - **Estimated Fraud Probability:** {score:.2%}
            - **Expected Loss Prevented:** ₹{exp_val:,.2f}
            - **Desk Review Expense:** ₹{review_cost:,.2f}
            - **Estimated Marginal ROI:** **₹{net_exp_val:,.2f}**
            - **Monthly Desk Quota:** 40 claims per month (approx. 1.3 claims/day)
            
            *Note on Economics:* In temporal validation (May–June 2026), blind top-40 monthly reviews yielded **8.75% precision** (₹13,131 fraud captured vs ₹30,400 review cost = -₹17,269 net value). The expected-value filter prevents reviewing low-dollar claims where the review fee exceeds potential savings.
            """)

        # Operational Disclaimer
        st.markdown(f"""
        <div class="disclaimer-box">
            <b>Operational Governance Notice:</b> {res['disclaimer']}
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("👈 Enter claim parameters in the sidebar or select a preset, then click **Assess Claim Risk**.")
