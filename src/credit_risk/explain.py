"""Explanations with SHAP: why the model gave an applicant the score it did.

SHAP splits one prediction into a contribution per feature. For tree models the
values are exact (TreeExplainer), and they add up:

    base value + sum of all contributions = the model's raw score (log-odds)

A positive contribution pushes the risk up, a negative one pulls it down. Listed in plain
language, they give reason-code-style explanations. They are not a compliant adverse-action
notice (ECOA / Regulation B): that would need reviewed reason wording, a stable mapping from
features to reasons, and legal review.

Caveats worth knowing: SHAP explains the *model*, not the real world (it is not
causal), and correlated features (the three credit scores, loan amount and goods
price) can share credit in ways that look arbitrary.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from credit_risk.config import MODEL_INPUTS

LABELS = {
    "amt_credit": "Loan amount",
    "amt_annuity": "Regular payment",
    "amt_goods_price": "Price of the purchase",
    "contract_type": "Loan type",
    "amt_income_total": "Income",
    "income_type": "Source of income",
    "education_type": "Education",
    "years_employed": "Years in current job",
    "ext_source_1": "Credit bureau score 1",
    "ext_source_2": "Credit bureau score 2",
    "ext_source_3": "Credit bureau score 3",
    "own_car": "Owns a car",
    "own_realty": "Owns a home",
    "cnt_children": "Children",
    "region_rating": "Region rating",
    "payment_to_income": "Payment-to-income",
    "credit_to_income": "Loan-to-income",
    "loan_to_value": "Loan-to-value",
    "credit_to_annuity": "Loan length",
    "not_employed": "Not employed",
}

MISSING_TEXT = {
    "ext_source_1": "no score on file",
    "ext_source_2": "no score on file",
    "ext_source_3": "no score on file",
    "years_employed": "not employed",
    "amt_goods_price": "not given",
}


def format_value(feature: str, value: Any) -> str:
    """The applicant's value for a feature, written the way a person would say it."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return MISSING_TEXT.get(feature, "missing")
    if feature.startswith("ext_source"):
        return f"{value:.2f}"
    if feature.startswith("amt_"):
        return f"{value:,.0f}"
    if feature in ("payment_to_income", "loan_to_value"):
        return f"{value:.0%}"
    if feature == "credit_to_income":
        return f"{value:.1f}× income"
    if feature == "credit_to_annuity":
        return f"{value:.0f} payments"
    if feature == "years_employed":
        return f"{value:.1f} years"
    if feature == "region_rating":
        return f"{int(value)} of 3"
    if feature == "cnt_children":
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    return str(value)


@dataclass(frozen=True)
class Reason:
    feature: str
    label: str
    value: str
    impact: float  # SHAP contribution in log-odds; positive = raises risk


class Explainer:
    """SHAP's TreeExplainer, wrapped around the model inside a fitted pipeline."""

    def __init__(self, pipeline: Pipeline):
        self.preprocess = pipeline[:-1]  # features + category casting, already fitted
        self.tree = shap.TreeExplainer(pipeline[-1])
        # The average raw score (log-odds) the contributions start from.
        self.base_value = float(np.ravel(self.tree.expected_value)[-1])

    def explain(self, applicants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (SHAP values, model features), one row per applicant, one column per feature."""
        features = self.preprocess.transform(applicants[MODEL_INPUTS])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # SHAP warns about a LightGBM output-format change
            values = self.tree.shap_values(features)
        if isinstance(values, list):  # older SHAP versions return one array per class
            values = values[1]
        return pd.DataFrame(values, columns=features.columns, index=features.index), features

    def reasons(self, applicant: pd.DataFrame, top: int = 3) -> tuple[list[Reason], list[Reason]]:
        """Top factors raising and lowering one applicant's risk, strongest first."""
        values, features = self.explain(applicant)
        contributions = values.iloc[0]
        row = features.iloc[0]

        def make(feature: str) -> Reason:
            return Reason(feature, LABELS.get(feature, feature), format_value(feature, row[feature]),
                          round(float(contributions[feature]), 4))

        ordered = contributions.sort_values()
        raising = [make(f) for f in ordered.index[::-1] if contributions[f] > 0.01][:top]
        lowering = [make(f) for f in ordered.index if contributions[f] < -0.01][:top]
        return raising, lowering


def global_importance(explainer: Explainer, applicants: pd.DataFrame) -> pd.Series:
    """Average size of each feature's contribution across many applicants (mean |SHAP|)."""
    values, _ = explainer.explain(applicants)
    return values.abs().mean().sort_values(ascending=False)


def plot_importance(explainer: Explainer, applicants: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Bar chart of global importance and a beeswarm plot (one dot per applicant per feature)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    values, features = explainer.explain(applicants)
    names = [LABELS.get(c, c) for c in features.columns]
    # The beeswarm colours dots by feature value (low = blue, high = red). Text categories
    # have no low or high, so they are left blank and SHAP draws them grey.
    numeric = features.apply(lambda c: np.nan if c.dtype == "category" else c.astype(float))

    paths = []
    for kind, filename in (("bar", "shap_importance.png"), ("dot", "shap_beeswarm.png")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-blank category columns, see above
            # rng: SHAP jitters overlapping dots randomly; a fixed seed keeps the figure reproducible.
            shap.summary_plot(values.to_numpy(), numeric, feature_names=names, plot_type=kind,
                              max_display=12, show=False, color="#2443d6" if kind == "bar" else None,
                              rng=np.random.default_rng(0))
        fig = plt.gcf()
        fig.axes[0].set_xlabel("Mean |SHAP| (log-odds)" if kind == "bar" else "SHAP contribution (log-odds)")
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)
        paths.append(out_dir / filename)
    return paths
