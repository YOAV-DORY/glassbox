"""Turning a probability of default into an approve/decline decision.

Portfolio simulation (approval rate, losses, profit across thresholds) comes in M4.
"""

from typing import Literal

Decision = Literal["approve", "decline"]


def cost_based_threshold(lgd: float, margin: float) -> float:
    """Highest probability of default at which approving still pays off.

    Approving a loan of size A earns margin * A if repaid and loses lgd * A on default.
    Expected profit (1 - p) * margin * A - p * lgd * A is positive exactly when
    p < margin / (margin + lgd). The loan size A cancels, so one threshold fits all loans.
    """
    if lgd <= 0 or margin <= 0:
        raise ValueError("lgd and margin must be positive")
    return margin / (margin + lgd)


def decide(p_default: float, threshold: float) -> Decision:
    return "approve" if p_default < threshold else "decline"
