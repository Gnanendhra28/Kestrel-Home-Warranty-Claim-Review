# Final Submission Checklist

**Project:** Kestrel Home — Warranty Claim Review (Variant C)  
**Date:** 5 October 2026  
**Status:** All Items Verified & Grounded  

---

| Item | Requirement | Verification Details | Status |
| :--- | :--- | :--- | :---: |
| **predictions.csv** | Official submission file exists | Located at `outputs/predictions.csv` | **[x] VERIFIED** |
| **Row Count** | Exactly 2,252 rows | Verified: exactly 2,252 data rows | **[x] VERIFIED** |
| **Format Match** | Exact match with `sample_submission.csv` | Columns: `claim_id,score`. Order identical from `WC711348` to `WC713599` | **[x] VERIFIED** |
| **FastAPI Endpoint** | Single-record JSON scoring service | Implemented in `src/api/main.py` (`POST /predict`, `GET /health`, `GET /examples`). Passes automated tests | **[x] VERIFIED** |
| **Streamlit Screen** | Working employee-facing review UI | Implemented in `app/app.py` (*"Fraud Risk Review"*). Connects via HTTP with automatic in-process fallback | **[x] VERIFIED** |
| **Human-Readable Reasons** | 3–5 evidence-based reasons per claim | Implemented in `src/business/risk_engine.py`. Evaluates 10 domain rules; zero "confirmed fraud" claims | **[x] VERIFIED** |
| **27+ Tests Passing** | Complete test suite passes | `pytest tests/ -v`: 27 passed, 0 failed in 1.81s | **[x] VERIFIED** |
| **No Paid API Dependency** | 100% local, zero paid cloud APIs | Runs entirely on open-source Scikit-Learn, FastAPI, Streamlit. Paid API cost = ₹0/month | **[x] VERIFIED** |
| **Business Memo** | One-page executive memo for Ritu Deshpande | Created at `outputs/reports/ritu_deshpande_memo.md` | **[x] VERIFIED** |
| **submission-form.md** | Complete answers to all 14 questions | Created at `submission-form.md` using only verified project facts | **[x] VERIFIED** |
| **Final Runbook** | Detailed operational guide | Created at `FINAL_RUNBOOK.md` | **[x] VERIFIED** |
| **Limitations Documented** | Realistic assessment of constraints | Documented in `outputs/reports/business_value_analysis.md`, memo, and runbook | **[x] VERIFIED** |
| **Source Data Protection** | Raw data preserved and unmodified | SHA-256 hashes verified via `test_data_integrity.py`. No raw data uploaded | **[x] VERIFIED** |
| **Recording Link Placeholder** | Explicitly marked if unavailable | Marked as: `"Not available yet — to be recorded and added by the candidate."` | **[x] VERIFIED** |
| **Google Drive Placeholder** | Explicitly marked if unavailable | Marked as: `"Not available yet — to be created and added by the candidate."` | **[x] VERIFIED** |
| **GitHub Repository** | Public repository URL | `https://github.com/Gnanendhra28/Kestrel-Home-Warranty-Claim-Review` | **[x] VERIFIED** |

---

## Verification Sign-Off
All 16 checklist items have been checked and confirmed. No further code modifications, model retraining, or external deployments are required.
