"""The shape of reports/metrics.json, the file the README and readers of the results rely on.

Training validates the report against this schema before writing it, and a test checks the
committed file, so a renamed or missing field fails loudly instead of silently breaking a reader.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelName = Literal["logreg", "lgbm"]
Share = Field(ge=0, le=1)


class ModelMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roc_auc: float = Share
    gini: float = Field(ge=-1, le=1)
    pr_auc: float = Share
    ks: float = Share
    brier: float = Share
    calibration_error: float = Share
    mean_predicted: float = Share
    actual_default_rate: float = Share


class Interval(BaseModel):
    estimate: float
    low: float
    high: float


class BootstrapReport(BaseModel):
    models: dict[Literal["a", "b"], ModelName]
    a: Interval
    b: Interval
    difference: Interval  # b minus a
    n_boot: int = Field(gt=0)


class CalibrationRow(BaseModel):
    decile: int = Field(ge=1, le=10)
    n: int = Field(gt=0)
    low: float = Share
    high: float = Share
    mean_predicted: float = Share
    actual_rate: float = Share
    actual_se: float = Field(ge=0)


class MetricsReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    selected_model: ModelName
    selection_rule: str
    lgbm_trees: int = Field(gt=0)
    validation: dict[ModelName, ModelMetrics]
    test: dict[ModelName, ModelMetrics]
    test_auc_bootstrap: BootstrapReport
    calibration_deciles: dict[Literal["validation", "test"], list[CalibrationRow]]
