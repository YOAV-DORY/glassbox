"""Model pipelines. Each pipeline takes the raw applicant columns and returns P(default).

Packing feature engineering, preprocessing and the model into one Pipeline means:
- everything learned from data (medians, scaling, categories) is fit on train only, so no leakage;
- one saved file contains everything the API needs to reproduce training exactly.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_risk.features import feature_builder

# Amounts have a long right tail (a few huge incomes). Log-scaling stops them from
# dominating a linear model.
AMOUNT_COLUMNS = ["amt_credit", "amt_annuity", "amt_goods_price", "amt_income_total"]
NUMERIC_COLUMNS = [
    "years_employed",
    "ext_source_1",
    "ext_source_2",
    "ext_source_3",
    "cnt_children",
    "region_rating",
    "payment_to_income",
    "credit_to_income",
    "loan_to_value",
    "credit_to_annuity",
]
BINARY_COLUMNS = ["own_car", "own_realty", "not_employed"]
CATEGORICAL_COLUMNS = ["contract_type", "income_type", "education_type"]


def build_logreg_pipeline() -> Pipeline:
    """Baseline: logistic regression on imputed, scaled, one-hot encoded features."""
    # add_indicator=True adds a 0/1 "was missing" column per feature with gaps. Missing
    # external scores are common (56% for ext_source_1), so "missing" must stay visible.
    impute_and_scale = [SimpleImputer(strategy="median", add_indicator=True), StandardScaler()]

    preprocess = ColumnTransformer(
        [
            ("amounts", make_pipeline(FunctionTransformer(np.log1p), *impute_and_scale), AMOUNT_COLUMNS),
            ("numeric", make_pipeline(*impute_and_scale), NUMERIC_COLUMNS),
            ("binary", "passthrough", BINARY_COLUMNS),
            # Categories seen fewer than 100 times in training (e.g. 5 people on maternity
            # leave) are grouped into one "infrequent" bucket instead of getting own weights.
            # Unseen categories at prediction time are ignored instead of crashing.
            (
                "categorical",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    # No class_weight="balanced": it would push predicted probabilities up and break their
    # meaning as default rates. Class imbalance is handled by the decision threshold instead.
    return Pipeline(
        [
            ("features", feature_builder()),
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
