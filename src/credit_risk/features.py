"""Engineered features, computed inside the model pipeline.

Because this step is part of the saved pipeline, the API computes the exact same
features as training did. Nothing here is learned from data, so it is a plain
function wrapped in a scikit-learn FunctionTransformer.
"""

import pandas as pd
from sklearn.preprocessing import FunctionTransformer

ENGINEERED_FEATURES = [
    "payment_to_income",
    "credit_to_income",
    "loan_to_value",
    "credit_to_annuity",
    "not_employed",
]


def add_engineered_features(applicants: pd.DataFrame) -> pd.DataFrame:
    df = applicants.copy()
    # Affordability, similar to a debt-to-income ratio. The dataset does not document
    # whether annuity and income are monthly or yearly, so treat ratios as relative signals.
    df["payment_to_income"] = df["amt_annuity"] / df["amt_income_total"]
    df["credit_to_income"] = df["amt_credit"] / df["amt_income_total"]
    # Loan amount vs price of the goods bought. Above 1 means borrowing more than the price.
    df["loan_to_value"] = df["amt_credit"] / df["amt_goods_price"]
    # Loan size measured in payments: a rough proxy for loan length.
    df["credit_to_annuity"] = df["amt_credit"] / df["amt_annuity"]
    # data.clean turned the "no employment" placeholder into missing, and the API sends
    # null for "not employed", so one rule covers both training and serving.
    df["not_employed"] = df["years_employed"].isna()
    return df


def feature_builder() -> FunctionTransformer:
    return FunctionTransformer(add_engineered_features, feature_names_out=_feature_names_out)


def _feature_names_out(transformer: FunctionTransformer, input_features: list[str]) -> list[str]:
    return [*input_features, *ENGINEERED_FEATURES]
