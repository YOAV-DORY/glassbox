# Glassbox

A credit-risk model you can see inside. Glassbox predicts loan default on the Home Credit
dataset and explains every decision with SHAP reason codes. It also includes a fairness
audit, a cost-based approval threshold, and a React + FastAPI decision app.

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
4. Put both in `data/raw/`. No need to unzip: `application_train.csv.zip` works as is.

Check it loaded:

```bash
python -m credit_risk.data
```
