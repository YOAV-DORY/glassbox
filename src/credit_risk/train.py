"""Train the model and save it for the API.

Run: python -m credit_risk.train

Only train and validation data are used here. The test split stays untouched until
the final evaluation, so its numbers are an honest estimate of real performance.
"""

import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk.artifacts import artifacts_dir, save_artifacts
from credit_risk.config import MODEL_INPUTS, PROJECT_ROOT, TARGET
from credit_risk.data import load_splits
from credit_risk.evaluate import basic_metrics
from credit_risk.models import build_logreg_pipeline


def train(train_df: pd.DataFrame) -> Pipeline:
    # Pass only the model inputs: protected attributes never reach the pipeline.
    pipeline = build_logreg_pipeline()
    pipeline.fit(train_df[MODEL_INPUTS], train_df[TARGET])
    return pipeline


def predict_default_proba(pipeline: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Column 1 of predict_proba is P(defaulted = 1)."""
    return pd.Series(pipeline.predict_proba(df[MODEL_INPUTS])[:, 1], index=df.index)


def git_commit() -> str:
    """Which code produced the model; "-dirty" means there were uncommitted changes."""
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return result.stdout.strip()

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    return f"{commit}-dirty" if git("status", "--porcelain") else commit


def build_metadata(model_type: str, train_rows: int, val_metrics: dict[str, float]) -> dict[str, Any]:
    trained_at = datetime.now(UTC)
    return {
        "model_version": f"{model_type}-{trained_at:%Y%m%d-%H%M%S}",
        "model_type": model_type,
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "inputs": MODEL_INPUTS,
        "train_rows": train_rows,
        "validation_metrics": {k: round(v, 4) for k, v in val_metrics.items()},
        # The saved pipeline is a pickle: it must be loaded with these same versions.
        "library_versions": {lib: version(lib) for lib in ("scikit-learn", "pandas", "numpy")},
    }


def main() -> None:
    splits = load_splits()
    pipeline = train(splits.train)

    val_metrics = basic_metrics(splits.val[TARGET], predict_default_proba(pipeline, splits.val))
    metadata = build_metadata("logreg", len(splits.train), val_metrics)
    out_dir = artifacts_dir()
    save_artifacts(pipeline, metadata, out_dir)

    print(f"Saved {metadata['model_version']} to {out_dir}")
    for name, value in val_metrics.items():
        print(f"  val {name:<20} {value:.4f}")


if __name__ == "__main__":
    main()
