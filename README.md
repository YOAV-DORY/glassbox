# Glassbox

**A credit-risk model you can see inside.** Glassbox predicts the probability that a loan applicant
defaults, turns it into an approve-or-decline decision with an explicit lending policy, and shows its
working: in a live decision app, and in a 3D view of the model's actual risk surface.

![Glassbox landing page](docs/images/landing.jpg)

> **Status:** work in progress. Data, model, evaluation, API and app are done (milestones M0–M2).
> Next: SHAP reason codes (M3), a policy and portfolio simulator (M4), and a fairness audit (M5).
> See [docs/PLAN.md](docs/PLAN.md).

## The problem

A lender receives an application and must say yes or no. Getting the probability right is only part
of the job:

1. **Mistakes don't cost the same.** Approving a loan that defaults can lose most of it; turning away
   someone who would have repaid loses only the margin. A 50% cut-off ignores that.
2. **Decisions must be explainable.** In the US, a lender that declines someone must give the principal
   reasons (ECOA / Regulation B adverse action notices).
3. **Decisions must be fair, and that has to be measured.** Leaving gender and age out of the model is
   not enough when other inputs can stand in for them.

Glassbox is built around those three points, on the public
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) data
(307,511 past applicants, 8.1% defaulted).

## What it does today

![The decision desk](docs/images/assess.jpg)

- **Model:** LightGBM on 15 application fields plus 5 engineered ratios (payment-to-income,
  loan-to-value, ...). A higher credit score can never raise predicted risk (monotone constraints).
- **Policy:** approve when expected profit is positive: `p < margin / (margin + loss given default)`.
  With placeholder economics (60% loss on default, 10% margin) the cut-off is **14.3%**.
- **API (FastAPI):** `POST /predict`, `GET /model/info`, `GET /model/surface`. Docs at `/docs`.
- **App (React + TypeScript):** a landing page and a live decision desk. The decision updates as
  you type; three example applicants (low risk, borderline just under the cut-off, high risk) are one click away.

## Results

Held-out **test set** (61,503 applicants), scored once, after the model was chosen on validation.

| Model | ROC-AUC | Gini | PR-AUC | KS | Brier | Calibration error |
|---|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.745 | 0.490 | 0.231 | 0.363 | 0.0685 | 0.3 pts |
| **LightGBM (served)** | **0.763** | **0.526** | **0.250** | **0.394** | **0.0676** | **0.3 pts** |

A random model scores ROC-AUC 0.5 and PR-AUC 0.081 (the default rate).

<p>
  <img src="reports/figures/calibration.png" width="49%" alt="Calibration curves">
  <img src="reports/figures/risk_map.png" width="49%" alt="Risk map of two credit scores with the cut-off">
</p>

- **Both models are well calibrated** (predicted vs. actual default rates within 0.3 points), so no
  extra calibration step is needed. This matters because the cut-off formula uses the probabilities
  directly.
- **The risk map** (right) is the served model with two credit scores varied and everything else held
  at a typical applicant. Steps come from tree thresholds; risk only falls towards higher scores
  (the monotone constraint); the dashed line is the policy cut-off.
- **20 features vs. all 117:** the curated set scored 0.758 validation ROC-AUC vs. 0.753 for every raw
  column, likely thanks to the engineered ratios
  ([notebook 02](notebooks/02_feature_set_tradeoff.ipynb)). The small set makes the form usable and the
  reasons readable, at no cost in accuracy.

More: [ROC](reports/figures/roc.png), [precision-recall](reports/figures/precision_recall.png),
[all metrics](reports/metrics.json), [exploratory analysis](notebooks/01_eda.ipynb).

## Design decisions

| Decision | Why |
|---|---|
| No accuracy metric | Approving everyone is 92% "accurate" and catches no defaults. |
| No class weights or resampling | They distort probabilities; imbalance is handled by the cut-off instead. |
| Early stopping on log loss, not AUC | Log loss rewards honest probabilities, not just a good ranking. |
| Monotone constraints on credit scores | Explanations stay intuitive; it's what model-risk reviewers expect. |
| Feature engineering inside the saved pipeline | Training and the API compute features identically. A test caught a real train/serve skew bug: a JSON `null` became a text column that LightGBM rejected. |
| Gender, age, marital status excluded from the model, kept for the audit | Mirrors US fair-lending rules. EDA found the "not employed" flag is mostly pensioners, i.e. an age proxy, so the audit has to look for proxies. |
| Model chosen on validation, test set scored once | Keeps the reported numbers honest. |

## Limitations

- **No time-based split.** The data has no application dates, so the split is random (stratified).
  In production I'd train on older loans, validate on newer ones, and monitor drift.
- **Selection bias.** Outcomes exist only for loans Home Credit approved. We never see how rejected
  applicants would have done, which also limits any fairness conclusion.
- **One lender, one market.** Not US data, consumer cash and revolving loans only; the currency and the payment period are
  undocumented, so ratios are relative signals.
- **Placeholder economics.** 60% loss and 10% margin are assumptions; real values vary per loan.
- **Opaque inputs.** The three external scores are the strongest features, and their provenance is unknown.
- **Not built yet:** reason codes (M3) and the fairness audit (M5).

## Run it locally

Needs Python 3.12+, Node 20+, and on macOS `brew install libomp` (for LightGBM).

1. **Data** (not in the repo; Kaggle's terms restrict redistribution): log in to Kaggle, open the
   [competition data page](https://www.kaggle.com/competitions/home-credit-default-risk/data), accept
   the rules, download `application_train.csv` and put it, zipped or not, in `data/raw/`.
2. **Install and train** (training takes about 10 seconds):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   python -m credit_risk.train
   ```
3. **Run the API and the app**, in two terminals:
   ```bash
   uvicorn api.main:app --reload
   ```
   ```bash
   cd web && npm install && npm run dev
   ```
   Open http://localhost:5173.

Tests need no data download: `pytest` (24 tests: cleaning and split, features, the monotone
constraint, metrics, threshold maths, and the API contract, all on synthetic data).

## Repository layout

```
src/credit_risk/   data.py, features.py, models.py, train.py, evaluate.py, decision.py, whatif.py
api/               FastAPI app and request/response schemas
web/               React + TypeScript app (Vite)
tests/             pytest suite (synthetic data)
notebooks/         01 exploratory analysis, 02 feature-set trade-off
reports/           metrics.json and figures (aggregates only)
docs/              plan and screenshots
```

Built with Python, pandas, scikit-learn, LightGBM, FastAPI, React and TypeScript. The 3D views are
drawn by a ~100-line canvas renderer (`web/src/wire3d.ts`), no 3D library.
