"""Shared test fixtures.

Tests use small synthetic data shaped like the Kaggle CSV, so they run on a fresh
clone without downloading anything.
"""

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import DAYS_EMPLOYED_SENTINEL


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
