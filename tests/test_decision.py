import pytest

from credit_risk.decision import cost_based_threshold, decide


def test_threshold_from_costs() -> None:
    # Lose 60% on a default, earn 10% on a repaid loan: approve below 0.1 / 0.7 = 1/7.
    assert cost_based_threshold(lgd=0.6, margin=0.1) == pytest.approx(1 / 7)
    # Equal costs give the familiar 0.5.
    assert cost_based_threshold(lgd=0.2, margin=0.2) == pytest.approx(0.5)


def test_threshold_rejects_non_positive_costs() -> None:
    with pytest.raises(ValueError):
        cost_based_threshold(lgd=0, margin=0.1)


def test_expected_profit_is_zero_at_the_threshold() -> None:
    lgd, margin = 0.6, 0.1
    p = cost_based_threshold(lgd, margin)
    assert (1 - p) * margin - p * lgd == pytest.approx(0)


def test_decide_approves_strictly_below_threshold() -> None:
    assert decide(0.10, threshold=0.14) == "approve"
    assert decide(0.14, threshold=0.14) == "decline"
    assert decide(0.30, threshold=0.14) == "decline"
