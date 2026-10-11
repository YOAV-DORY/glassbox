"""Model metrics and evaluation plots.

Accuracy is deliberately absent: with 8% defaults, approving everyone scores 92%
accuracy while catching no defaults.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

INK, COBALT, MUTED = "#15171c", "#2443d6", "#8a8d94"


def ks_statistic(y_true: np.ndarray, p_default: np.ndarray) -> float:
    """Kolmogorov-Smirnov: the largest gap between the score distributions of defaulters
    and repayers. Equals the maximum of (true positive rate - false positive rate)."""
    fpr, tpr, _ = roc_curve(y_true, p_default)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y_true: np.ndarray, p_default: np.ndarray, n_bins: int = 10) -> float:
    """Average gap between predicted and actual default rates, over 10 equal-sized groups
    of applicants sorted by prediction. 0.01 means predictions are off by ~1 point."""
    bins = pd.qcut(p_default, n_bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y_true, "p": p_default, "bin": bins})
    per_bin = df.groupby("bin").agg(actual=("y", "mean"), predicted=("p", "mean"), n=("y", "size"))
    return float(np.average((per_bin["actual"] - per_bin["predicted"]).abs(), weights=per_bin["n"]))


def calibration_table(y_true: np.ndarray, p_default: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Applicants sorted by prediction and cut into 10 equal-sized groups (deciles).

    For each: the range and average of predictions, the actual default rate, and that rate's
    standard error. A single average error can hide a miss near the cut-off; this table can't.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p_default)})
    df["decile"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop") + 1
    table = df.groupby("decile").agg(
        n=("y", "size"), low=("p", "min"), high=("p", "max"), mean_predicted=("p", "mean"), actual_rate=("y", "mean")
    )
    table["actual_se"] = np.sqrt(table["actual_rate"] * (1 - table["actual_rate"]) / table["n"])
    return [
        {"decile": int(decile), "n": int(row["n"]), **{k: round(float(row[k]), 5) for k in
                                                     ("low", "high", "mean_predicted", "actual_rate", "actual_se")}}
        for decile, row in table.iterrows()
    ]


def basic_metrics(y_true: np.ndarray, p_default: np.ndarray) -> dict[str, float]:
    y_true, p_default = np.asarray(y_true), np.asarray(p_default)
    auc = roc_auc_score(y_true, p_default)
    return {
        "roc_auc": auc,  # ranking: P(random defaulter scores higher than random repayer)
        "gini": 2 * auc - 1,  # same information, the scale lenders usually quote
        "pr_auc": average_precision_score(y_true, p_default),  # random model = default rate
        "ks": ks_statistic(y_true, p_default),
        "brier": brier_score_loss(y_true, p_default),  # mean squared error of probabilities
        "calibration_error": expected_calibration_error(y_true, p_default),
        "mean_predicted": float(np.mean(p_default)),  # should be close to the actual rate
        "actual_default_rate": float(np.mean(y_true)),
    }


def plot_evaluation(y_true: np.ndarray, predictions: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    """ROC, precision-recall and calibration curves, one line per model. Returns file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = [COBALT, INK]
    base_rate = float(np.mean(y_true))
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for (name, p), color in zip(predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", label="Random")
    ax.set(xlabel="False positive rate (repayers flagged)", ylabel="True positive rate (defaulters caught)",
           title="ROC curve (test set)")
    ax.legend(frameon=False, loc="lower right")
    paths.append(_save(fig, out_dir / "roc.png"))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for (name, p), color in zip(predictions.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, p)
        ax.plot(recall, precision, color=color,
                label=f"{name} (PR-AUC {average_precision_score(y_true, p):.3f})")
    ax.axhline(base_rate, color=MUTED, linestyle="--", label=f"Random ({base_rate:.1%})")
    ax.set(xlabel="Recall (share of defaulters caught)", ylabel="Precision (share of flags that default)",
           title="Precision-recall curve (test set)")
    ax.legend(frameon=False, loc="upper right")
    paths.append(_save(fig, out_dir / "precision_recall.png"))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    upper = 0.0
    for (name, p), color in zip(predictions.items(), colors):
        actual, predicted = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        ax.plot(predicted, actual, marker="o", markersize=4, color=color, label=name)
        upper = max(upper, actual.max(), predicted.max())
    upper *= 1.1  # zoom to where the groups are, instead of the full 0-1 range
    ax.plot([0, upper], [0, upper], color=MUTED, linestyle="--", label="Perfectly calibrated")
    ax.set(xlim=(0, upper), ylim=(0, upper))
    ax.set(xlabel="Predicted default rate", ylabel="Actual default rate",
           title="Calibration (test set, 10 equal-sized groups)")
    ax.legend(frameon=False, loc="upper left")
    paths.append(_save(fig, out_dir / "calibration.png"))
    return paths


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
