"""Train both models, pick the better one, evaluate it, and save it for the API.

Run: python -m credit_risk.train

Order matters for honest numbers:
1. Fit on train (LightGBM also uses validation for early stopping).
2. Choose the model on validation.
3. Only then score the test set, once, for the final report.
"""

import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk.artifacts import artifacts_dir, save_artifacts
from credit_risk.config import DEFAULT_LGD, DEFAULT_MARGIN, MODEL_INPUTS, PROJECT_ROOT, REPORTS_DIR, TARGET
from credit_risk.data import load_splits
from credit_risk.evaluate import basic_metrics, plot_evaluation
from credit_risk.decision import cost_based_threshold
from credit_risk.models import build_logreg_pipeline, fit_lgbm
from credit_risk.whatif import plot_risk_map

MODEL_NAMES = {"logreg": "Logistic regression", "lgbm": "LightGBM"}


def train_logreg(train_df: pd.DataFrame) -> Pipeline:
    # Pass only the model inputs: protected attributes never reach the pipeline.
    pipeline = build_logreg_pipeline()
    pipeline.fit(train_df[MODEL_INPUTS], train_df[TARGET])
    return pipeline


def predict_default_proba(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Column 1 of predict_proba is P(defaulted = 1)."""
    return pipeline.predict_proba(df[MODEL_INPUTS])[:, 1]


def reference_applicant(train_df: pd.DataFrame) -> dict[str, Any]:
    """A 'typical' applicant: median of each number, most common value of everything else.
    Used as the fixed background when the app varies one or two inputs at a time."""
    reference: dict[str, Any] = {}
    for col in MODEL_INPUTS:
        values = train_df[col].dropna()
        if pd.api.types.is_bool_dtype(values) or not pd.api.types.is_numeric_dtype(values):
            reference[col] = values.mode().iloc[0]
        else:
            reference[col] = values.median()
    # JSON can't store numpy types; .item() turns them into plain Python numbers/bools.
    return {k: v.item() if hasattr(v, "item") else v for k, v in reference.items()}


def git_commit() -> str:
    """Which code produced the model; "-dirty" means there were uncommitted changes."""
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return result.stdout.strip()

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    return f"{commit}-dirty" if git("status", "--porcelain") else commit


def rounded(metrics: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in metrics.items()}


def build_metadata(
    model_type: str,
    train_df: pd.DataFrame,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    trained_at = datetime.now(UTC)
    return {
        "model_version": f"{model_type}-{trained_at:%Y%m%d-%H%M%S}",
        "model_type": model_type,
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "inputs": MODEL_INPUTS,
        "train_rows": len(train_df),
        "validation_metrics": rounded(val_metrics),
        "test_metrics": rounded(test_metrics) if test_metrics else None,
        "reference_applicant": reference_applicant(train_df),
        # The saved pipeline is a pickle: it must be loaded with these same versions.
        "library_versions": {lib: version(lib) for lib in ("scikit-learn", "lightgbm", "pandas", "numpy")},
    }


def main() -> None:
    splits = load_splits()
    y_val, y_test = splits.val[TARGET].to_numpy(), splits.test[TARGET].to_numpy()

    print("Training logistic regression and LightGBM...")
    models = {"logreg": train_logreg(splits.train), "lgbm": fit_lgbm(splits.train, splits.val)}
    val_metrics = {name: basic_metrics(y_val, predict_default_proba(m, splits.val)) for name, m in models.items()}
    best = max(val_metrics, key=lambda name: val_metrics[name]["roc_auc"])

    # The model is chosen. Now, and only now, look at the test set.
    test_predictions = {name: predict_default_proba(m, splits.test) for name, m in models.items()}
    test_metrics = {name: basic_metrics(y_test, p) for name, p in test_predictions.items()}
    figures = plot_evaluation(
        y_test, {MODEL_NAMES[name]: p for name, p in test_predictions.items()}, REPORTS_DIR / "figures"
    )

    report = {
        "selected_model": best,
        "selection_rule": "highest validation ROC-AUC; test set scored once, after selection",
        "lgbm_trees": int(models["lgbm"][-1].best_iteration_),
        "validation": {name: rounded(m) for name, m in val_metrics.items()},
        "test": {name: rounded(m) for name, m in test_metrics.items()},
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))

    metadata = build_metadata(best, splits.train, val_metrics[best], test_metrics[best])
    save_artifacts(models[best], metadata, artifacts_dir())
    figures.append(plot_risk_map(
        models[best], metadata["reference_applicant"], cost_based_threshold(DEFAULT_LGD, DEFAULT_MARGIN),
        REPORTS_DIR / "figures" / "risk_map.png",
    ))

    table = pd.DataFrame({f"{MODEL_NAMES[n]} ({split})": m for split, ms in
                          (("val", val_metrics), ("test", test_metrics)) for n, m in ms.items()})
    print(table.round(4).to_string())
    print(f"\nLightGBM stopped at {report['lgbm_trees']} trees.")
    print(f"Selected {MODEL_NAMES[best]} -> saved {metadata['model_version']} to {artifacts_dir()}")
    print("Figures:", ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in figures))


if __name__ == "__main__":
    main()
