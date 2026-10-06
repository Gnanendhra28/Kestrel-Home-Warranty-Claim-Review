# Executive Memorandum

**TO:** Ritu Deshpande, Head of D2C Operations  
**FROM:** Lead AI/ML Engineer  
**DATE:** 5 October 2026  
**SUBJECT:** Warranty Fraud Claim Review — Operational Decision Recommendation  

---

### DECISION: What Kestrel Should Do Now
Transition our warranty investigation desk from blind/manual sampling to a **risk-ranked, value-weighted decision support queue**. 

Crucially, **do not mandate a flat 40-claim monthly audit quota**. Under current operating conditions, reviewing low-value claims costs more in customer goodwill delays (₹380/claim) than the fraud prevented. Instead, use the model's continuous risk score to investigate **only claims where the expected fraud value exceeds the ₹380 review cost** ($P(\text{fraud}) \times \text{Claim Amount} > ₹380$). All other claims should proceed along standard settlement workflows.

---

### WHAT THE DATA SHOWS
1. **The Sub-₹2,000 Shift:** Following the 1 May 2026 policy change introducing fast-track approval for claims under ₹2,000, claim behavior altered drastically. In our May–June validation data, **97.2% of confirmed fraud claims were submitted just below ₹2,000** (average claim amount: ₹1,864).
2. **Partner Risk is Behavioral, Not Age-Based:** While earlier exploratory cuts suggested newer service centers had higher fraud incidence, temporal auditing revealed that partner onboarding stopped mid-2025. In current operations (Q2 FY26), all partners have matured beyond 365 days of tenure. Fraud correlates with **velocity spikes, serial re-use, and repair price ratios**, not partner tenure.
3. **The Accuracy Mirage:** A naive model that approves 100% of claims automatically achieves **97.47% accuracy** simply because confirmed fraud prevalence is low (~2.5%). The board's >97% accuracy KPI is therefore not a meaningful operational standard. What matters is ranking quality and net recovery.

---

### ₹ IMPACT: Modeled Economics on Validation Data
Across the 2-month validation horizon (May–June 2026; 1,422 claims, 36 ground-truth frauds):
- **Frauds Captured:** The model identified **7 confirmed frauds** in the top 40 monthly reviews (19.4% recall; 8.75% precision), stopping **₹13,131** in fraudulent payouts (+₹9,886 over random baseline).
- **Goodwill Review Expense:** Auditing 80 claims at the policy friction cost of ₹380/claim totaled **₹30,400**.
- **Modeled Net Value:** **-₹17,269** (a net loss of ₹215.86 per investigated claim).
- **Break-Even Reality:** At an average fraud size of ₹1,864, the desk requires a precision of **~20.4%** ($\frac{₹380}{₹1,864}$) to break even. A blind 40-claims/month review cannot achieve positive ROI on micro-claims. Value-weighted triage fixes this by prioritizing high-ticket exposures (e.g. ₹5,000+ claims where 25% risk yields >₹1,250 expected savings).

---

### RISKS & LIMITATIONS
- **Low Fraud Base Rate:** Confirmed fraud accounts for only 2.5% of intake, making uncalibrated manual triage costly.
- **Micro-Claim Exploitation:** Auto-approving sub-₹2,000 claims creates a systemic incentive for low-dollar claim padding.
- **Legacy Zoho Uncertainty:** Pre-migration CRM records stored unresolved investigations as `0` rather than null, creating potential label noise in older historical records.
- **Zero Cost for Inference:** The decision-support pipeline runs 100% locally with open-source models, adding **₹0/month** in third-party API or LLM fees.

---

### NEXT WEEK: Three Practical Operational Actions
1. **Deploy Value-Weighted Review Queue:** Configure the triage screen so investigators only receive claims where expected fraud savings exceed ₹380, preventing wasteful audits on low-dollar tickets.
2. **Reinstate Mandatory Photo Proof Below ₹2,000:** Update portal validation rules to require photographic evidence for all parts-replacement claims, closing the auto-approval loophole without delaying legitimate customers.
3. **Establish a Weekly Audit Feedback Loop:** Ingest newly completed investigation findings into the Empirical Bayes partner tracker every Monday to keep risk priors aligned with emerging partner behavior.
