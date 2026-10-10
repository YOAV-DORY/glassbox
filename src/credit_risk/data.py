"""Load, clean and split the Home Credit application data.

Run `python -m credit_risk.data` for a quick summary of the dataset and the split.
"""

from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import (
    ALL_RENAMES,
    AUDIT_COLUMNS,
    DAYS_EMPLOYED_SENTINEL,
    DAYS_PER_YEAR,
    ID_COL,
    MODEL_INPUTS,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
    TRAIN_CSV,
    VAL_SIZE,
)


class Splits(NamedTuple):
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_raw(path: Path = TRAIN_CSV) -> pd.DataFrame:
    """Read only the columns we use (~20 of 122) from the Kaggle CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download application_train.csv from "
            "https://www.kaggle.com/competitions/home-credit-default-risk/data "
            "and put it in data/raw/ (see README)."
        )
    return pd.read_csv(path, usecols=list(ALL_RENAMES))


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw Kaggle columns into a tidy applicant table.

    This only fixes the *format* of the data (names, units, placeholder values).
    Anything learned from data (imputation, scaling, ratios) belongs in the model
    pipeline instead, so it is fit on the training split only.
    """
    if not raw["SK_ID_CURR"].is_unique:
        raise ValueError("Duplicate applicant IDs: a person could land in both train and test.")

    df = raw.rename(columns=ALL_RENAMES)

    # Days are stored as negative numbers counted back from the application date.
    # The employment sentinel becomes missing; the pipeline later adds a
    # "not employed" flag from that missingness, using the same rule for API input.
    days_employed = df["years_employed"].mask(df["years_employed"] == DAYS_EMPLOYED_SENTINEL)
    df["years_employed"] = -days_employed / DAYS_PER_YEAR
    df["age_years"] = -df["age_years"] / DAYS_PER_YEAR

    df["own_car"] = df["own_car"].eq("Y")
    df["own_realty"] = df["own_realty"].eq("Y")

    # 4 applicants have gender "XNA" (unknown). Keep them for training, but leave
    # gender missing so the fairness audit skips them.
    df["gender"] = df["gender"].mask(df["gender"] == "XNA")

    return df[[ID_COL, TARGET, *MODEL_INPUTS, *AUDIT_COLUMNS]]


def split(df: pd.DataFrame, seed: int = RANDOM_SEED) -> Splits:
    """Stratified 60/20/20 split, so each part keeps the ~8% default rate.

    The data has no application date, so a time-based split is not possible.
    In production you would train on older loans and test on newer ones.
    """
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=seed
    )
    # VAL_SIZE is a share of the full data, so rescale it to the remaining 80%.
    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_val[TARGET],
        random_state=seed,
    )
    return Splits(train, val, test)


def load_splits(path: Path = TRAIN_CSV, seed: int = RANDOM_SEED) -> Splits:
    return split(clean(load_raw(path)), seed=seed)


def main() -> None:
    df = clean(load_raw())
    print(f"Applicants: {len(df):,}   Columns used: {df.shape[1]}")
    print(f"Default rate: {df[TARGET].mean():.2%}")
    print(f"Not employed (sentinel -> missing): {df['years_employed'].isna().sum():,}")

    print("\nMissing values in model inputs:")
    missing = df[MODEL_INPUTS].isna().mean().sort_values(ascending=False)
    print(missing[missing > 0].map("{:.1%}".format).to_string())

    print("\nSplit sizes and default rates:")
    for name, part in split(df)._asdict().items():
        print(f"  {name:<5} {len(part):>7,}  default rate {part[TARGET].mean():.2%}")


if __name__ == "__main__":
    main()
