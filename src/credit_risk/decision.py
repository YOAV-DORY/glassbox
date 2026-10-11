"""Turning probabilities of default into decisions, and decisions into money.

Economics (per loan of size A, as shares of A):
- approve and repaid:   earn margin * A          (a false rejection would lose this)
- approve and defaults: lose lgd * A             (lgd = loss given default; the cost of a false approval)
- decline:              nothing happens

Simplifications, stated up front: one period, no time value of money, the same lgd and
margin for every loan, and the whole loan amount is at risk on default.
"""

from typing import Literal

import numpy as np

Decision = Literal["approve", "decline"]


def cost_based_threshold(lgd: float, margin: float) -> float:
    """Highest probability of default at which approving still pays off.

    Expected profit (1 - p) * margin * A - p * lgd * A is positive exactly when
    p < margin / (margin + lgd). The loan size A cancels, so one threshold fits all loans.
    This is only right if p is calibrated: an honest probability, not just a ranking.
    """
    if lgd <= 0 or margin <= 0:
        raise ValueError("lgd and margin must be positive")
    return margin / (margin + lgd)


def decide(p_default: float, threshold: float) -> Decision:
    return "approve" if p_default < threshold else "decline"


def expected_profit(p_default: float, amount: float, lgd: float, margin: float) -> float:
    """Expected profit of approving one loan, before knowing how it turns out."""
    return (1 - p_default) * margin * amount - p_default * lgd * amount


def portfolio_outcome(
    p_default: np.ndarray,
    defaulted: np.ndarray,
    amount: np.ndarray,
    threshold: float,
    lgd: float,
    margin: float,
) -> dict[str, float]:
    """What a cut-off would have done to a set of past loans whose outcomes we know.

    "Expected" figures use only the model's probabilities (what you'd forecast before lending);
    "realized" figures use what actually happened. Close agreement = calibrated in money terms.

    Selection bias: these are loans the lender already approved. Declining one of them here is a
    what-if on an approved loan, so decline-side results are assumptions, not measurements of
    how genuinely rejected applicants would have behaved.
    """
    approved = p_default < threshold
    n_approved = int(approved.sum())
    p, a = p_default[approved], amount[approved]
    repaid = approved & (defaulted == 0)
    bad = approved & (defaulted == 1)
    realized_loss = float((amount[bad] * lgd).sum())
    return {
        "threshold": threshold,
        "approval_rate": n_approved / len(p_default),
        "bad_rate": float(defaulted[approved].mean()) if n_approved else 0.0,  # defaults among approved
        "approved": n_approved,
        "expected_loss": float((p * lgd * a).sum()),
        "realized_loss": realized_loss,
        "expected_profit": float(((1 - p) * margin * a - p * lgd * a).sum()),
        "realized_profit": float((amount[repaid] * margin).sum()) - realized_loss,
    }


def profit_curve(
    p_default: np.ndarray,
    defaulted: np.ndarray,
    amount: np.ndarray,
    lgd: float,
    margin: float,
    thresholds: np.ndarray,
) -> list[dict[str, float]]:
    """Portfolio outcome at each candidate cut-off, for plotting profit against the cut-off."""
    return [portfolio_outcome(p_default, defaulted, amount, float(t), lgd, margin) for t in thresholds]
