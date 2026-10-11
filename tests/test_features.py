import numpy as np
import pandas as pd
import pytest

from credit_risk.config import MODEL_INPUTS
from credit_risk.features import add_engineered_features
from credit_risk.train import train_logreg
from credit_risk.data import clean
from tests.conftest import VALID_APPLICANT, make_raw_df


def test_engineered_ratios_and_not_employed_flag() -> None:
    applicants = pd.DataFrame(
        [
            {**VALID_APPLICANT, "amt_credit": 400, "amt_annuity": 20, "amt_goods_price": 200,
             "amt_income_total": 100, "years_employed": 3.0},
            {**VALID_APPLICANT, "amt_goods_price": None, "years_employed": None},
        ]
    )
    out = add_engineered_features(applicants)

    first = out.iloc[0]
    assert first["payment_to_income"] == pytest.approx(0.2)
    assert first["credit_to_income"] == pytest.approx(4.0)
    assert first["loan_to_value"] == pytest.approx(2.0)
    assert first["credit_to_annuity"] == pytest.approx(20.0)
    assert not first["not_employed"]

    second = out.iloc[1]
    assert np.isnan(second["loan_to_value"])  # missing goods price stays missing, no crash
    assert second["not_employed"]


def test_pipeline_handles_missing_scores_and_rare_categories() -> None:
    pipeline = train_logreg(clean(make_raw_df(n=2000)))
    # "Student" never appears in the synthetic training data, and all scores are missing.
    unusual = {**VALID_APPLICANT, "income_type": "Student", "ext_source_1": None,
               "ext_source_2": None, "ext_source_3": None, "years_employed": None}
    rows = pd.DataFrame([VALID_APPLICANT, unusual])[MODEL_INPUTS]

    p_default = pipeline.predict_proba(rows)[:, 1]
    assert ((p_default > 0) & (p_default < 1)).all()
