# Explainable Credit Risk (working title)

A default-prediction model on the Home Credit dataset, with SHAP reason codes, a
fairness audit, a cost-based approval threshold, and a React + FastAPI decision app.

> Work in progress. See [docs/PLAN.md](docs/PLAN.md) for the design and milestones.
> The full README (results, fairness findings, limitations) arrives in the final milestone.

## Setup

Requirements: Python 3.12+, Node 20+, and on macOS `libomp` for LightGBM (`brew install libomp`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Data

The data is not in this repo (Kaggle's terms restrict redistribution).

1. Log in at [kaggle.com](https://www.kaggle.com) and open the
   [Home Credit Default Risk data page](https://www.kaggle.com/competitions/home-credit-default-risk/data).
2. Accept the competition rules.
3. Download `application_train.csv` and `HomeCredit_columns_description.csv`.
4. Put both in `data/raw/`.

Check it loaded:

```bash
python -m credit_risk.data
```
