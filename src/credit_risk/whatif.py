"""What-if analysis: how predicted risk changes when we vary inputs on a fixed applicant."""

from typing import Any

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
