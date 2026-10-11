"""Model metrics. Accuracy is deliberately absent: with 8% defaults, approving everyone
scores 92% accuracy while catching no defaults."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def basic_metrics(y_true: np.ndarray, p_default: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y_true, p_default)
    return {
        "roc_auc": auc,  # ranking: P(random defaulter scores higher than random payer)
        "gini": 2 * auc - 1,  # same information, the scale lenders usually quote
        "pr_auc": average_precision_score(y_true, p_default),  # random model = default rate
        "brier": brier_score_loss(y_true, p_default),  # mean squared error of probabilities
        "mean_predicted": float(np.mean(p_default)),  # should be close to the actual rate
        "actual_default_rate": float(np.mean(y_true)),
    }
