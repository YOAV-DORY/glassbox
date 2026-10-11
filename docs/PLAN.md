# Project plan

The approved Phase 1 plan, kept in the repo as a record of the design decisions.

## Decisions

| Question | Decision |
|---|---|
| Dataset | Home Credit Default Risk, `application_train.csv` only |
| Protected attributes in the model | Excluded: gender, age, marital status (kept for the fairness audit only) |
| Frontend language | TypeScript |
| Python environment | `venv` + `pyproject.toml` with exact pins |
| Default policy assumptions | Loss given default 60%, net margin 10% (placeholders, labeled as such) |
| Repo | `glassbox`, public on GitHub |

## 1. Dataset

Home Credit has gender and age, which makes the fairness audit possible. Lending Club
does not (US lenders mostly cannot collect them) and has many post-origination columns
that leak the outcome.

Costs of this choice:
- No application date, so no out-of-time split. We use a stratified random split and state it as a limitation.
- Not US data, and a single lender's consumer loans. The currency is unspecified.
- We skip the 6 other tables. Top Kaggle scores (~0.80 AUC) used them; the main table alone gives roughly 0.74–0.76.

## 2. Stack

Python 3.13: pandas 3.0, numpy 2.5, scikit-learn 1.9, LightGBM 4.7, SHAP 0.53,
FastAPI 0.143, Pydantic 2.14, pytest 9.1. Frontend: Vite + React 19 + TypeScript,
Recharts, plain CSS. The Vite dev server proxies `/api` to FastAPI, so no CORS setup.

LightGBM over XGBoost: fast, handles missing values and categoricals natively, which
gives one SHAP value per original feature and cleaner reason codes.

## 3. Modeling

**Features.** 15 raw applicant fields a person can fill in a form, plus 5 engineered ones
computed inside the pipeline: payment-to-income, credit-to-income, loan-to-value,
approximate term, and a not-employed flag.

Raw Kaggle columns are renamed and converted to readable units in `data.clean`, so the
training table has exactly the same columns as an API request. The saved pipeline then
sees identical inputs at training and serving time (no train/serve skew).

**Split.** Stratified 60/20/20. Train fits everything, validation is for early stopping and
calibration, test is used once for final numbers, the portfolio simulation and fairness.
All preprocessing lives inside a scikit-learn `Pipeline` fit on train only.

**Models.** Logistic regression baseline, LightGBM main model. No resampling or class
weights, so predicted probabilities stay calibrated; the threshold handles imbalance.

**Metrics.** ROC-AUC and Gini, PR-AUC, KS, Brier score and a calibration curve.
Accuracy is misleading: approving everyone scores ~92% accuracy and catches no defaults.

## 4. Explainability

SHAP `TreeExplainer` (exact for trees, milliseconds per applicant). Global importance
plots, and per-applicant reason codes: top factors raising and lowering risk. Test:
SHAP values plus the base value add up to the model's raw log-odds.

## 5. Fairness

Groups: gender and age bands (<30, 30–44, 45–59, 60+). Per group at the chosen
threshold: base default rate, approval rate, adverse impact ratio (flag below 0.8),
false rejection rate, false approval rate, calibration, AUC. Optional proxy test:
predict gender from model features. Caveats: different base rates make all fairness
criteria impossible to satisfy at once; labels exist only for previously approved loans.

## 6. Cost-based threshold

Approve when expected profit is positive: `(1 - p) * margin - p * LGD > 0`, which gives
`p < margin / (margin + LGD)`. Portfolio view on the test set: approval rate, bad rate,
expected and realized loss, profit, compared with "approve everyone" and a 0.5 cutoff.

## 7. API

- `GET /health`
- `GET /model/info`: metrics, global importance, categorical options
- `POST /predict`: applicant + optional policy -> PD, decision, threshold, expected profit, reasons
- `POST /portfolio/simulate`: policy -> portfolio KPIs, profit curve, fairness tables

## 8. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Setup and data | `python -m credit_risk.data` prints 307,511 rows, ~8.1% default; tests pass |
| M1 | Thin end-to-end slice (logistic regression, `/predict`, minimal form) | Browser form returns PD and decision from the saved model |
| M2 | LightGBM and evaluation | Metrics and calibration report; API serves LightGBM |
| M3 | SHAP explanations | Additivity test passes; reasons show in the UI |
| M4 | Cost-based threshold | Threshold tests pass; policy sliders update live |
| M5 | Fairness audit | Group tables and written findings |
| M6 | README and interview notes | Fresh clone runs in 5 commands or fewer |
| M7 | Stretch: Docker and deploy | Deployed without shipping row-level data |
