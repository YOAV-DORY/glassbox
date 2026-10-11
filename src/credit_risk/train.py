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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk.artifacts import artifacts_dir, save_artifacts, save_cohort
from credit_risk.config import (
    DEFAULT_LGD,
    DEFAULT_MARGIN,
    MODEL_INPUTS,
    PROJECT_ROOT,
    REPORTS_DIR,
    TARGET,
    VALIDATION_COHORT_FILE,
)
from credit_risk.data import clean, load_raw, split
from credit_risk.decision import cost_based_threshold
from credit_risk.evaluate import basic_metrics, calibration_table, paired_bootstrap_auc, plot_evaluation
from credit_risk.explain import LABELS, Explainer, global_importance, plot_importance
from credit_risk.fairness import audit
from credit_risk.models import build_logreg_pipeline, fit_lgbm
from credit_risk.reports import MetricsReport
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


def cohort_frame(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """One row per applicant: the model's prediction, what really happened, the loan amount,
    and the audit-only columns (gender, age) plus the 'not employed' flag for the proxy check."""
    return pd.DataFrame({
        "p_default": predict_default_proba(pipeline, df),
        "defaulted": df[TARGET].to_numpy(),
        "amt_credit": df["amt_credit"].to_numpy(),
        "gender": df["gender"].to_numpy(),
        "age_years": df["age_years"].to_numpy(),
        "not_employed": df["years_employed"].isna().to_numpy(),
    })


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
        try:
            result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
        except FileNotFoundError:  # no git installed (e.g. inside the Docker image)
            return ""
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


def run(
    raw: pd.DataFrame,
    out_dir: Path,
    reports_dir: Path,
    dataset: str,
    n_boot: int = 1000,
    verbose: bool = True,
) -> dict[str, Any]:
    """The whole training protocol, from raw Kaggle-format rows to saved model and reports.

    Used for the real data (main), the synthetic demo (credit_risk.demo) and the end-to-end
    test, so all three follow exactly the same steps.
    """
    say = print if verbose else (lambda *args: None)
    splits = split(clean(raw))
    y_val, y_test = splits.val[TARGET].to_numpy(), splits.test[TARGET].to_numpy()
    figures_dir = reports_dir / "figures"
    threshold = cost_based_threshold(DEFAULT_LGD, DEFAULT_MARGIN)

    say("Training logistic regression and LightGBM...")
    models = {"logreg": train_logreg(splits.train), "lgbm": fit_lgbm(splits.train, splits.val)}
    val_predictions = {name: predict_default_proba(m, splits.val) for name, m in models.items()}
    val_metrics = {name: basic_metrics(y_val, p) for name, p in val_predictions.items()}
    best = max(val_metrics, key=lambda name: val_metrics[name]["roc_auc"])

    # The model is chosen. Now, and only now, look at the test set.
    test_predictions = {name: predict_default_proba(m, splits.test) for name, m in models.items()}
    test_metrics = {name: basic_metrics(y_test, p) for name, p in test_predictions.items()}
    figures = plot_evaluation(y_test, {MODEL_NAMES[name]: p for name, p in test_predictions.items()}, figures_dir)

    report = {
        "dataset": dataset,
        "selected_model": best,
        "selection_rule": "highest validation ROC-AUC; test set scored once, after selection",
        "lgbm_trees": int(models["lgbm"][-1].best_iteration_),
        "validation": {name: rounded(m) for name, m in val_metrics.items()},
        "test": {name: rounded(m) for name, m in test_metrics.items()},
        # How much the test AUCs would move with a different sample of applicants (reporting only:
        # the model was already chosen on validation, and nothing is refitted).
        "test_auc_bootstrap": {
            "models": {"a": "logreg", "b": "lgbm"},
            **paired_bootstrap_auc(y_test, test_predictions["logreg"], test_predictions["lgbm"], n_boot=n_boot),
        },
        # Predicted vs actual default rate per decile, for the selected model. Validation is what
        # the simulator works on; test is the final check, scored once like everything else here.
        "calibration_deciles": {
            "validation": calibration_table(y_val, val_predictions[best]),
            "test": calibration_table(y_test, test_predictions[best]),
        },
    }
    MetricsReport.model_validate(report)  # fail here, not in whatever reads the file later
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    metadata = build_metadata(best, splits.train, val_metrics[best], test_metrics[best])
    metadata["dataset"] = dataset
    ci = report["test_auc_bootstrap"]["b" if best == "lgbm" else "a"]
    metadata["test_roc_auc_ci"] = [ci["low"], ci["high"]]
    if best == "lgbm":  # SHAP's TreeExplainer needs a tree model
        explainer = Explainer(models[best])
        sample = splits.test.sample(min(10_000, len(splits.test)), random_state=0)  # more rows add little
        importance = global_importance(explainer, sample)
        metadata["global_importance"] = [
            {"feature": f, "label": LABELS[f], "importance": round(float(v), 4)} for f, v in importance.items()
        ]
        figures += plot_importance(explainer, sample, figures_dir)
    save_artifacts(models[best], metadata, out_dir)

    # Policy and fairness analysis run on validation: users explore it interactively, and
    # nothing interactive may touch the test set, which is kept for the headline metrics.
    validation_cohort = cohort_frame(models[best], splits.val)
    save_cohort(validation_cohort, out_dir, VALIDATION_COHORT_FILE)
    fairness = audit(validation_cohort, threshold)
    fairness["cohort"] = "validation"
    (reports_dir / "fairness.json").write_text(json.dumps(fairness, indent=2, default=float))
    figures.append(plot_risk_map(models[best], metadata["reference_applicant"], threshold, figures_dir / "risk_map.png"))

    table = pd.DataFrame({f"{MODEL_NAMES[n]} ({split_name})": m for split_name, ms in
                          (("val", val_metrics), ("test", test_metrics)) for n, m in ms.items()})
    say(table.round(4).to_string())
    say(f"\nLightGBM stopped at {report['lgbm_trees']} trees.")
    say(f"Selected {MODEL_NAMES[best]} -> saved {metadata['model_version']} to {out_dir}")
    say("Figures:", ", ".join(str(p) for p in figures))
    return report


def main() -> None:
    run(load_raw(), artifacts_dir(), REPORTS_DIR, dataset="home-credit")


if __name__ == "__main__":
    main()
