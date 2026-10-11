"""Fairness audit: how the model and the lending policy treat groups the model never sees.

Gender and age are not model inputs. They were kept aside in data.clean only for this audit.
Leaving them out does not make the model fair by itself ("fairness through unawareness"):
other inputs can stand in for them, which is why we measure outcomes per group.

What the numbers can and cannot say:
- Different approval rates can come from different default rates (base rates) rather than
  from the model treating similar people differently. Both columns are reported side by side.
- When base rates differ, no model can equalise approval rates, error rates and calibration
  all at once; choosing which matters is a business and legal decision, not a modelling one.
- Outcomes exist only for loans the lender approved in the past (selection bias).
"""

from typing import Any

import numpy as np
import pandas as pd

AGE_BINS = [0, 30, 45, 60, 200]
AGE_LABELS = ["<30", "30-44", "45-59", "60+"]


def age_band(age_years: pd.Series) -> pd.Series:
    return pd.cut(age_years, bins=AGE_BINS, right=False, labels=AGE_LABELS)


def group_report(cohort: pd.DataFrame, groups: pd.Series, threshold: float) -> pd.DataFrame:
    """One row per group at a given cut-off.

    `cohort` needs columns p_default (model output) and defaulted (real outcome).
    Rows with an unknown group (e.g. gender "XNA") are left out.
    """
    df = cohort.assign(group=groups, approved=cohort["p_default"] < threshold).dropna(subset=["group"])
    rows = []
    for name, g in df.groupby("group", observed=True):
        approved, repayers = g[g["approved"]], g[g["defaulted"] == 0]
        rows.append({
            "group": str(name),
            "n": len(g),
            "default_rate": float(g["defaulted"].mean()),  # base rate: what actually happened
            # Sampling noise on that rate (binomial standard error): gaps of several SEs are real.
            "default_rate_se": float(np.sqrt(g["defaulted"].mean() * (1 - g["defaulted"].mean()) / len(g))),
            "mean_predicted": float(g["p_default"].mean()),  # close to default_rate = calibrated for this group
            "approval_rate": float(g["approved"].mean()),
            "default_rate_approved": float(approved["defaulted"].mean()) if len(approved) else float("nan"),
            # Equal-opportunity view: of the people who did repay, how many would we have turned away?
            "repayers_declined": float((~repayers["approved"]).mean()),
        })
    report = pd.DataFrame(rows)
    # Adverse impact ratio: each group's approval rate relative to the most-approved group.
    # Below 0.8 is the common "four-fifths" warning sign (a rule of thumb, not a legal test).
    report["approval_ratio"] = report["approval_rate"] / report["approval_rate"].max()
    return report


def proxy_check(cohort: pd.DataFrame) -> dict[str, float]:
    """How strongly the model's 'not employed' input stands in for age.

    In this data the employment placeholder marks pensioners, so the flag is close to "60+".
    """
    old = cohort["age_years"] >= 60
    flagged = cohort["not_employed"].astype(bool)
    return {
        "share_60plus_not_employed": float(flagged[old].mean()),
        "share_under_45_not_employed": float(flagged[cohort["age_years"] < 45].mean()),
        "share_not_employed_who_are_60plus": float(old[flagged].mean()) if flagged.any() else float("nan"),
    }


def audit(cohort: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """The full audit at one cut-off, as plain data (for reports/fairness.json and the API)."""

    def records(report: pd.DataFrame) -> list[dict[str, Any]]:
        return [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
                for row in report.to_dict("records")]

    return {
        "threshold": threshold,
        "gender": records(group_report(cohort, cohort["gender"], threshold)),
        "age_band": records(group_report(cohort, age_band(cohort["age_years"]), threshold)),
        "proxy": proxy_check(cohort),
    }
