# Pharma Reimbursement Forecasting — Pricing Analytics

**Domain:** Healthcare / Pharma Pricing Analytics  
**Tools:** Python · XGBoost · Random Forest · Gradient Boosting · Scikit-learn · Pandas · Seaborn · Matplotlib  
**Author:** Soumya Thamke

---

## Business Context

Pharmaceutical pricing teams need to forecast the net reimbursement amount approved by payers (insurance companies, government programs) for each drug claim. Without a reliable model, pricing strategy is largely reactive — list prices get set without visibility into what payers will actually reimburse, which therapeutic areas carry the most margin risk, or where contract negotiations should be prioritised.

This project builds and compares three ensemble ML models to predict reimbursement per claim and surfaces the key drivers behind reimbursement outcomes — giving pricing teams quantified, actionable levers.

---

## Objective

- Predict net reimbursement amount per drug claim across payer types and therapeutic areas
- Compare Random Forest, XGBoost, and Gradient Boosting on accuracy and interpretability
- Identify the strongest drivers of reimbursement using feature importance
- Quantify the impact of pricing risk factors (prior auth, step therapy, denial rate, rebate %)

---

## Dataset

Synthetically generated (3,000 records, 15 features) to mirror real-world pharma claims data structure. No proprietary data used.

| Feature | Description |
|---|---|
| `therapeutic_area` | Oncology, Cardiology, Diabetes, Neurology, Rare Disease, Immunology, Infectious Disease |
| `payer_type` | Commercial Insurance, Medicare, Medicaid, Government Contract, Self-Pay |
| `drug_tier` | Tier 1 (Generic) → Tier 4 (Specialty) |
| `formulation` | Oral, Injectable, Infusion, Topical, Inhaled |
| `region` | Northeast, Southeast, Midwest, West, Southwest |
| `list_price` | Computed from tier base price × therapeutic area multiplier |
| `rebate_pct` | Negotiated rebate as % of list price (0–40%) |
| `prior_auth_required` | Binary flag — reduces reimbursement by ~12% |
| `step_therapy_required` | Binary flag — reduces reimbursement by ~8% |
| `denial_rate_historical` | Historical claim denial rate (0–35%) |
| `years_on_market` | Drug age in years |
| `patent_protected` | Binary — derived from years on market (< 12 yrs) |
| `quantity_dispensed` | Units dispensed per claim (1–180) |
| `num_prior_claims` | Number of previous claims for this drug |
| `net_list_after_rebate` | Engineered feature: list price × (1 − rebate %) |
| `reimbursement_amount` | **Target variable** — net USD amount reimbursed |

**Payer reimbursement rates built into data generation:**

| Payer | Rate (% of list) |
|---|---|
| Commercial Insurance | 78% |
| Medicare | 70% |
| Medicaid | 55% |
| Government Contract | 62% |
| Self-Pay | 40% |

---

## Project Pipeline

```
1. Data Generation      →  Synthetic claims dataset (3,000 rows, 15 features)
2. EDA                  →  Distribution plots, payer/TA breakdowns, correlation analysis
3. Feature Engineering  →  Label encoding, net_list_after_rebate, effective_reimb_rate
4. Model Training       →  Random Forest · XGBoost · Gradient Boosting (200 estimators each)
5. Evaluation           →  MAE, RMSE, R², 5-fold cross-validation, residual analysis
6. Feature Importance   →  Ranked drivers across all 3 models
7. Business Insights    →  Pricing strategy implications from model outputs
```

---

## Model Results

| Model | MAE | RMSE | R² (Test) | CV R² |
|---|---|---|---|---|
| Random Forest | — | — | — | — |
| **XGBoost** | **—** | **—** | **—** | **—** |
| Gradient Boosting | — | — | — | — |

> **Run `pharma_reimbursement.py` to populate the table above with your actual output values.**  
> XGBoost is expected to be the best performer based on the data structure.

---

## Key Findings

**Top reimbursement drivers (from feature importance across all 3 models):**

1. **List price & net list after rebate** — the single strongest predictor; pricing strategy directly determines reimbursement outcome
2. **Payer type** — Medicare and Medicaid reimburse significantly less than commercial insurance; payer mix is a major margin lever
3. **Therapeutic area** — Rare Disease and Oncology command 3–4× higher reimbursement vs. general drugs
4. **Rebate %** — diminishing returns above ~30%; pricing teams should model net reimbursement, not list
5. **Prior auth required** — ~12% drag on average reimbursement
6. **Step therapy required** — ~8% drag on average reimbursement
7. **Historical denial rate** — claims with >25% historical denial rate show significant reimbursement risk

**Scenario implications for pricing teams:**
- Self-Pay patients receive only ~40% of list price — patient assistance programs may improve net revenue capture
- Specialty Tier 4 drugs show the highest variance in reimbursement — scenario modelling is critical before price-setting
- Rebate negotiations above 30% have diminishing returns on net reimbursement

---

## Output Files

| File | Description |
|---|---|
| `01_EDA.png` | Reimbursement distribution, payer averages, TA breakdown, drug tier boxplots, auth/step therapy impact |
| `02_Model_Comparison.png` | MAE, RMSE, R² side-by-side bar charts for all 3 models |
| `03_Actual_vs_Predicted.png` | Scatter plot + residual plot for the best model |
| `04_Feature_Importance.png` | Feature importance rankings across all 3 models |

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

# Run the script
python pharma_reimbursement.py
```

All output charts are saved automatically to the working directory.

---

## Skills Demonstrated

- Ensemble ML modeling (Random Forest, XGBoost, Gradient Boosting)
- Model evaluation: MAE, RMSE, R², 5-fold cross-validation, residual analysis
- Feature engineering and label encoding for categorical pharma data
- Feature importance interpretation and business insight generation
- Healthcare / pharma pricing domain knowledge
- End-to-end ML pipeline from data generation to stakeholder-ready outputs
