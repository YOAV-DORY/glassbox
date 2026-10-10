import numpy as np
import pandas as pd
import pytest

from credit_risk.config import AUDIT_COLUMNS, ID_COL, MODEL_INPUTS, TARGET
from credit_risk.data import clean, split


def test_clean_produces_model_and_audit_columns(raw_df: pd.DataFrame) -> None:
    df = clean(raw_df)
    assert list(df.columns) == [ID_COL, TARGET, *MODEL_INPUTS, *AUDIT_COLUMNS]


def test_clean_converts_days_to_positive_years_and_sentinel_to_missing(
    raw_df: pd.DataFrame,
) -> None:
    df = clean(raw_df)
    assert df["years_employed"].isna().sum() == 100  # every 10th of 1000 rows was the sentinel
    assert (df["years_employed"].dropna() > 0).all()
    assert df["age_years"].between(21, 69).all()
    assert df.loc[1, "years_employed"] == pytest.approx(-raw_df.loc[1, "DAYS_EMPLOYED"] / 365.25)


def test_clean_maps_flags_and_unknown_gender(raw_df: pd.DataFrame) -> None:
    df = clean(raw_df)
    assert df["own_car"].dtype == bool
    assert (df["own_car"] == raw_df["FLAG_OWN_CAR"].eq("Y")).all()
    assert pd.isna(df.loc[0, "gender"])  # "XNA" in the fixture


def test_clean_rejects_duplicate_ids(raw_df: pd.DataFrame) -> None:
    raw_df.loc[1, "SK_ID_CURR"] = raw_df.loc[0, "SK_ID_CURR"]
    with pytest.raises(ValueError, match="Duplicate"):
        clean(raw_df)


def test_split_is_60_20_20_stratified_disjoint_and_reproducible(raw_df: pd.DataFrame) -> None:
    df = clean(raw_df)
    splits = split(df, seed=1)

    sizes = [len(part) for part in splits]
    assert sizes == [600, 200, 200]

    ids = [set(part[ID_COL]) for part in splits]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])

    # Stratification keeps the default rate close to the overall rate in every part.
    overall = df[TARGET].mean()
    for part in splits:
        assert part[TARGET].mean() == pytest.approx(overall, abs=0.01)

    again = split(df, seed=1)
    assert np.array_equal(splits.test[ID_COL].to_numpy(), again.test[ID_COL].to_numpy())
