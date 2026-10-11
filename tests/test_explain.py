import numpy as np
import pandas as pd
import pytest

from credit_risk.config import MODEL_INPUTS
from credit_risk.explain import Explainer, format_value
from tests.conftest import VALID_APPLICANT


@pytest.fixture(scope="module")
def explainer(lgbm_pipeline) -> Explainer:
    return Explainer(lgbm_pipeline)


def test_contributions_add_up_to_the_model_score(lgbm_pipeline, explainer) -> None:
    """The defining property of SHAP: base value + contributions = raw score (log-odds)."""
    applicants = pd.DataFrame([VALID_APPLICANT, {**VALID_APPLICANT, "ext_source_2": 0.05, "years_employed": None}])
    values, features = explainer.explain(applicants[MODEL_INPUTS])
    raw_score = lgbm_pipeline[-1].predict(features, raw_score=True)
    np.testing.assert_allclose(explainer.base_value + values.sum(axis=1), raw_score, atol=1e-6)


def test_reasons_have_the_right_signs_and_order(explainer) -> None:
    up, down = explainer.reasons(pd.DataFrame([{**VALID_APPLICANT, "ext_source_2": 0.02}]))
    assert all(r.impact > 0 for r in up) and all(r.impact < 0 for r in down)
    assert [r.impact for r in up] == sorted((r.impact for r in up), reverse=True)  # strongest first
    assert [r.impact for r in down] == sorted(r.impact for r in down)
    # In the synthetic data a low ext_source_2 means high risk, so it should be a top reason.
    assert up[0].feature == "ext_source_2" and up[0].label == "Credit bureau score 2"


@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        ("ext_source_1", np.nan, "no score on file"),
        ("years_employed", np.nan, "not employed"),
        ("amt_credit", 675000.0, "675,000"),
        ("payment_to_income", 0.208, "21%"),
        ("own_car", True, "yes"),
        ("region_rating", 2.0, "2 of 3"),
    ],
)
def test_values_are_written_for_people(feature: str, value: object, expected: str) -> None:
    assert format_value(feature, value) == expected
