import numpy as np
import pandas as pd
import pytest

from credit_risk.config import MODEL_INPUTS
from credit_risk.evaluate import calibration_table, expected_calibration_error, ks_statistic
from tests.conftest import VALID_APPLICANT


def test_lgbm_risk_never_rises_with_a_better_credit_score(lgbm_pipeline) -> None:
    """The monotone constraint: sweep ext_source_2 from 0 to 1 for one applicant."""
    grid = pd.DataFrame([{**VALID_APPLICANT, "ext_source_2": s} for s in np.linspace(0, 1, 41)])
    p_default = lgbm_pipeline.predict_proba(grid[MODEL_INPUTS])[:, 1]
    assert np.all(np.diff(p_default) <= 1e-12)


def test_lgbm_handles_unseen_category_and_missing_values(lgbm_pipeline) -> None:
    unusual = {**VALID_APPLICANT, "income_type": "Maternity leave", "ext_source_2": None,
               "years_employed": None, "amt_goods_price": None}
    p_default = lgbm_pipeline.predict_proba(pd.DataFrame([unusual])[MODEL_INPUTS])[:, 1]
    assert 0 < p_default[0] < 1


def test_ks_statistic_extremes() -> None:
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)  # perfect separation
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)  # no separation


def test_calibration_error_is_small_for_honest_probabilities() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.4, 200_000)
    y = (rng.random(p.size) < p).astype(int)  # outcomes drawn from the predicted probabilities
    assert expected_calibration_error(y, p) < 0.005
    assert expected_calibration_error(y, np.clip(p * 2, 0, 1)) > 0.1  # doubled predictions are off


def test_calibration_table_has_ten_ordered_deciles_that_match_honest_probabilities() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.4, 100_000)
    y = (rng.random(p.size) < p).astype(int)
    table = calibration_table(y, p)
    assert [row["decile"] for row in table] == list(range(1, 11))
    assert sum(row["n"] for row in table) == p.size
    assert all(a["high"] <= b["low"] for a, b in zip(table, table[1:]))  # deciles don't overlap
    for row in table:  # honest probabilities: each decile's actual rate within ~4 standard errors
        assert abs(row["actual_rate"] - row["mean_predicted"]) < 4 * row["actual_se"]
