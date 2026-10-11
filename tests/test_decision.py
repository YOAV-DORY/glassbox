import numpy as np
import pytest

from credit_risk.decision import (
    cost_based_threshold,
    decide,
    expected_profit,
    portfolio_outcome,
    profit_curve,
)


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


# --- Portfolio maths -------------------------------------------------------------


def test_portfolio_outcome_on_four_loans_by_hand() -> None:
    p = np.array([0.05, 0.10, 0.30, 0.02])
    defaulted = np.array([0, 1, 1, 0])
    amount = np.array([100.0, 200.0, 300.0, 400.0])
    out = portfolio_outcome(p, defaulted, amount, threshold=0.2, lgd=0.5, margin=0.1)
    # Approved: loans 1, 2 and 4 (p < 0.2). Loan 2 defaulted.
    assert out["approval_rate"] == pytest.approx(3 / 4)
    assert out["bad_rate"] == pytest.approx(1 / 3)
    assert out["realized_loss"] == pytest.approx(0.5 * 200)
    assert out["realized_profit"] == pytest.approx(0.1 * (100 + 400) - 0.5 * 200)
    assert out["expected_loss"] == pytest.approx(0.5 * (0.05 * 100 + 0.10 * 200 + 0.02 * 400))
    # Expected profit uses only the probabilities: (1 - p) * margin * A - p * lgd * A per approved loan.
    assert out["expected_profit"] == pytest.approx(
        sum(expected_profit(pi, ai, 0.5, 0.1) for pi, ai in [(0.05, 100), (0.10, 200), (0.02, 400)])
    )


def test_expected_profit_matches_its_definition() -> None:
    assert expected_profit(0.1, amount=1000, lgd=0.6, margin=0.1) == pytest.approx(0.9 * 100 - 0.1 * 600)


def test_formula_cut_off_lands_on_the_profit_peak_when_probabilities_are_honest() -> None:
    """The theory check: with calibrated probabilities, margin / (margin + lgd) maximises profit."""
    rng = np.random.default_rng(0)
    p = rng.beta(1.2, 10, 300_000)  # skewed towards low risk, like real applicants
    defaulted = (rng.random(p.size) < p).astype(int)  # outcomes drawn from the probabilities
    amount = rng.uniform(50_000, 1_000_000, p.size)
    lgd, margin = 0.6, 0.1

    thresholds = np.arange(0.01, 0.61, 0.01)
    curve = profit_curve(p, defaulted, amount, lgd, margin, thresholds)
    best = thresholds[int(np.argmax([point["realized_profit"] for point in curve]))]
    assert best == pytest.approx(cost_based_threshold(lgd, margin), abs=0.02)
