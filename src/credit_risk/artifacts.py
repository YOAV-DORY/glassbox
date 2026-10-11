"""Saving and loading the trained pipeline plus its metadata.

Training and the API both go through these two functions, so they always agree on
the file format. Only load model files you created yourself: unpickling runs code.
"""

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk.config import ARTIFACTS_DIR, METADATA_FILE, MODEL_FILE, VALIDATION_COHORT_FILE


def artifacts_dir() -> Path:
    """Default artifacts folder; tests point the API elsewhere via an environment variable."""
    return Path(os.environ.get("GLASSBOX_ARTIFACTS_DIR", ARTIFACTS_DIR))


def save_artifacts(pipeline: Pipeline, metadata: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / MODEL_FILE)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))


def load_artifacts(in_dir: Path) -> tuple[Pipeline, dict[str, Any]]:
    model_path = in_dir / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Train first: python -m credit_risk.train")
    return joblib.load(model_path), json.loads((in_dir / METADATA_FILE).read_text())


def save_cohort(predictions: pd.DataFrame, out_dir: Path, filename: str = VALIDATION_COHORT_FILE) -> None:
    """Save per-applicant predictions and outcomes for one data split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out_dir / filename, index=False)


def load_cohort(in_dir: Path, filename: str = VALIDATION_COHORT_FILE) -> pd.DataFrame | None:
    """Saved predictions for one split, or None if training hasn't produced them yet."""
    path = in_dir / filename
    return pd.read_parquet(path) if path.exists() else None
