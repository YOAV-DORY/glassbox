"""What-if analysis: how predicted risk changes when we vary inputs on a fixed applicant."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk.config import MODEL_INPUTS


def risk_surface(
    pipeline: Pipeline,
    reference: dict[str, Any],
    x_feature: str,
    y_feature: str,
    steps: int = 21,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Predicted default probability over a grid of two features (each from 0 to 1),
    with every other input held at the reference ("typical") applicant's value.

    Returns (x values, y values, z) where z[i][j] is the probability at y[i], x[j].
    One batch prediction of steps * steps rows: milliseconds, not a training run.
    """
    values = np.linspace(0, 1, steps)
    grid = pd.DataFrame(
        [{**reference, x_feature: x, y_feature: y} for y in values for x in values]
    )
    p_default = pipeline.predict_proba(grid[MODEL_INPUTS])[:, 1]
    z = p_default.reshape(steps, steps).round(4).tolist()
    return values.round(3).tolist(), values.round(3).tolist(), z


def plot_risk_map(
    pipeline: Pipeline, reference: dict[str, Any], threshold: float, path: Path, steps: int = 101
) -> Path:
    """The same surface as the app's 3D view, as a flat map: colour = predicted default risk,
    dashed line = the policy cut-off. Left of / below the line the policy declines."""
    x, y, z = risk_surface(pipeline, reference, "ext_source_2", "ext_source_3", steps)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(np.array(z) * 100, origin="lower", extent=(0, 1, 0, 1), cmap="magma_r",
                      vmin=0, vmax=40, aspect="auto")
    contour = ax.contour(x, y, z, levels=[threshold], colors="#2443d6", linestyles="--", linewidths=1.6)
    ax.clabel(contour, fmt={threshold: f"cut-off {threshold:.1%}"}, fontsize=9)
    fig.colorbar(image, ax=ax, label="Predicted default risk (%, capped at 40)")
    ax.set(xlabel="External credit score 2", ylabel="External credit score 3",
           title="Risk map: two scores varied, all else typical")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
