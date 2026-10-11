"""Synthetic applicants in the same raw format as Kaggle's application_train.csv.

Used for the demo (so the app runs without downloading the Kaggle data) and for end-to-end
tests. The numbers are invented: the shapes roughly follow the real data, and default risk is
driven by a hidden "creditworthiness" that the external scores partly reveal, so the model has
real signal to learn. Nothing here should be read as a finding about real borrowers.
"""

import numpy as np
import pandas as pd

from credit_risk.config import DAYS_EMPLOYED_SENTINEL


def make_synthetic_raw(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def pick(options: dict[str, float]) -> np.ndarray:
        return rng.choice(list(options), size=n, p=list(options.values()))

    income_type = pick({"Working": 0.52, "Commercial associate": 0.23, "Pensioner": 0.18, "State servant": 0.07})
    pensioner = income_type == "Pensioner"
    age = np.where(pensioner, rng.uniform(55, 69, n), rng.uniform(21, 60, n))
    gender = pick({"F": 0.66, "M": 0.34})

    # Hidden creditworthiness: older and female applicants slightly better on average,
    # loosely mirroring the base-rate differences in the real data.
    worth = rng.normal(0, 1, n) + 0.02 * (age - 43) - 0.2 * (gender == "M")

    def score(noise: float, missing: float) -> np.ndarray:
        s = 1 / (1 + np.exp(-(worth + rng.normal(0, noise, n))))
        return np.where(rng.random(n) < missing, np.nan, s)

    income = rng.lognormal(np.log(150_000), 0.45, n)
    credit = rng.lognormal(np.log(500_000), 0.6, n)
    n_payments = rng.uniform(10, 36, n)
    annuity = credit / n_payments
    days_employed = np.where(pensioner, DAYS_EMPLOYED_SENTINEL, -rng.exponential(6 * 365.25, n).astype(int) - 30)
    region = pick({"1": 0.10, "2": 0.74, "3": 0.16}).astype(int)
    own_realty = rng.random(n) < 0.69

    payment_to_income = annuity / income
    # Real credit risk has thresholds and interactions (which is why trees beat linear models
    # on the real data), so the invented risk has some too: an affordability cliff, a U-shape
    # in loan length, and weak scores hurting more in poorer regions.
    logit = (-3.45 - 0.9 * worth
             + 1.1 * (payment_to_income > 0.25) + 0.6 * ((n_payments < 13) | (n_payments > 32))
             + 0.8 * ((worth < -0.8) & (region == 3)) + 0.7 * (worth < -1.5)
             - 0.2 * own_realty)
    defaulted = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame({
        "SK_ID_CURR": np.arange(100_000, 100_000 + n),
        "TARGET": defaulted,
        "AMT_CREDIT": credit.round(0),
        "AMT_ANNUITY": annuity.round(1),
        "AMT_GOODS_PRICE": np.where(rng.random(n) < 0.001, np.nan, (credit / rng.uniform(1.0, 1.2, n)).round(0)),
        "NAME_CONTRACT_TYPE": pick({"Cash loans": 0.9, "Revolving loans": 0.1}),
        "AMT_INCOME_TOTAL": income.round(0),
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": pick({
            "Secondary / secondary special": 0.71, "Higher education": 0.24, "Incomplete higher": 0.03,
            "Lower secondary": 0.013, "Academic degree": 0.007,
        }),
        "DAYS_EMPLOYED": days_employed,
        "EXT_SOURCE_1": score(0.9, 0.56),
        "EXT_SOURCE_2": score(0.7, 0.002),
        "EXT_SOURCE_3": score(0.8, 0.20),
        "FLAG_OWN_CAR": np.where(rng.random(n) < 0.34, "Y", "N"),
        "FLAG_OWN_REALTY": np.where(own_realty, "Y", "N"),
        "CNT_CHILDREN": rng.poisson(0.4, n),
        "REGION_RATING_CLIENT": region,
        "CODE_GENDER": gender,
        "DAYS_BIRTH": -(age * 365.25).astype(int),
        "NAME_FAMILY_STATUS": pick({"Married": 0.64, "Single / not married": 0.15, "Civil marriage": 0.10,
                                    "Separated": 0.06, "Widow": 0.05}),
    })
