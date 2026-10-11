import numpy as np
import pandas as pd
import pytest

from credit_risk.fairness import age_band, audit, group_report, proxy_check


@pytest.fixture
def cohort() -> pd.DataFrame:
    # Six applicants, hand-made so every number below can be checked by eye.
    return pd.DataFrame({
        "p_default": [0.05, 0.10, 0.30, 0.02, 0.20, 0.08],
        "defaulted": [0, 1, 1, 0, 0, 0],
        "gender": ["F", "F", "F", "M", "M", None],  # the last one is unknown and must be skipped
        "age_years": [25.0, 35.0, 62.0, 50.0, 70.0, 40.0],
        "not_employed": [False, False, True, False, True, False],
    })


def test_group_report_by_hand(cohort: pd.DataFrame) -> None:
    report = group_report(cohort, cohort["gender"], threshold=0.15).set_index("group")
    assert list(report.index) == ["F", "M"]

    women = report.loc["F"]  # p = .05, .10, .30 -> approved, approved, declined
    assert women["n"] == 3
    assert women["default_rate"] == pytest.approx(2 / 3)
    assert women["default_rate_se"] == pytest.approx(np.sqrt((2 / 3) * (1 / 3) / 3))
    assert women["approval_rate"] == pytest.approx(2 / 3)
    assert women["default_rate_approved"] == pytest.approx(1 / 2)  # of the two approved, one defaulted
    assert women["repayers_declined"] == pytest.approx(0)  # the only repayer was approved

    men = report.loc["M"]  # p = .02, .20 -> approved, declined; both repaid
    assert men["approval_rate"] == pytest.approx(1 / 2)
    assert men["repayers_declined"] == pytest.approx(1 / 2)
    # Adverse impact ratio is relative to the most-approved group (women here).
    assert men["approval_ratio"] == pytest.approx((1 / 2) / (2 / 3))
    assert women["approval_ratio"] == pytest.approx(1.0)


def test_age_bands_cover_the_boundaries() -> None:
    bands = age_band(pd.Series([29.9, 30.0, 44.9, 45.0, 59.9, 60.0])).astype(str).tolist()
    assert bands == ["<30", "30-44", "30-44", "45-59", "45-59", "60+"]


def test_proxy_check(cohort: pd.DataFrame) -> None:
    proxy = proxy_check(cohort)
    assert proxy["share_60plus_not_employed"] == pytest.approx(1.0)  # both 60+ are flagged
    assert proxy["share_not_employed_who_are_60plus"] == pytest.approx(1.0)
    assert proxy["share_under_45_not_employed"] == pytest.approx(0.0)


def test_audit_is_plain_json_ready_data(cohort: pd.DataFrame) -> None:
    result = audit(cohort, threshold=0.01)  # nobody approved: rates among approved are undefined
    assert result["gender"][0]["default_rate_approved"] is None  # NaN would not be valid JSON
    assert {row["group"] for row in result["age_band"]} <= {"<30", "30-44", "45-59", "60+"}
    assert not any(isinstance(v, float) and np.isnan(v) for row in result["gender"] for v in row.values())
