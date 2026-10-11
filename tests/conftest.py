"""Shared test fixtures.

Tests use small synthetic data shaped like the Kaggle CSV, so they run on a fresh
clone without downloading anything.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.artifacts import save_artifacts
from credit_risk.config import DAYS_EMPLOYED_SENTINEL
from credit_risk.data import clean
from credit_risk.train import build_metadata, train


def make_raw_df(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Fake rows with the same raw columns and value formats as application_train.csv."""
    rng = np.random.default_rng(seed)
    days_employed = -rng.integers(30, 15000, size=n)
    days_employed[::10] = DAYS_EMPLOYED_SENTINEL  # every 10th applicant "not employed"
    gender = rng.choice(["F", "M"], size=n).astype(object)  # object: "<U1" would truncate "XNA"
    gender[0] = "XNA"

    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100000, 100000 + n),
            "TARGET": (rng.random(n) < 0.08).astype(int),
            "AMT_CREDIT": rng.uniform(50_000, 2_000_000, n),
            "AMT_ANNUITY": rng.uniform(2_000, 100_000, n),
            "AMT_GOODS_PRICE": rng.uniform(40_000, 1_800_000, n),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], size=n),
            "AMT_INCOME_TOTAL": rng.uniform(30_000, 500_000, n),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "State servant"], size=n),
            "NAME_EDUCATION_TYPE": rng.choice(
                ["Secondary / secondary special", "Higher education"], size=n
            ),
            "DAYS_EMPLOYED": days_employed,
            "EXT_SOURCE_1": np.where(rng.random(n) < 0.5, np.nan, rng.random(n)),
            "EXT_SOURCE_2": rng.random(n),
            "EXT_SOURCE_3": np.where(rng.random(n) < 0.2, np.nan, rng.random(n)),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], size=n),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], size=n),
            "CNT_CHILDREN": rng.integers(0, 4, size=n),
            "REGION_RATING_CLIENT": rng.integers(1, 4, size=n),
            "CODE_GENDER": gender,
            "DAYS_BIRTH": -rng.integers(21 * 365, 69 * 365, size=n),
            "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married"], size=n),
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_df()


# One applicant as the API receives it (and as the pipeline expects it after data.clean).
VALID_APPLICANT: dict = {
    "amt_credit": 450_000,
    "amt_annuity": 24_000,
    "amt_goods_price": 405_000,
    "contract_type": "Cash loans",
    "amt_income_total": 180_000,
    "income_type": "Working",
    "education_type": "Higher education",
    "years_employed": 4.5,
    "ext_source_1": None,
    "ext_source_2": 0.56,
    "ext_source_3": 0.41,
    "own_car": False,
    "own_realty": True,
    "cnt_children": 1,
    "region_rating": 2,
}


@pytest.fixture(scope="session")
def trained_artifacts_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real pipeline trained on synthetic data and saved like the train script does."""
    out_dir = tmp_path_factory.mktemp("artifacts")
    df = clean(make_raw_df(n=2000))
    pipeline = train(df)
    save_artifacts(pipeline, build_metadata("logreg", len(df), {"roc_auc": 0.5}), out_dir)
    return out_dir
