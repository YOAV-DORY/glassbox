"""Model pipelines. Each pipeline takes the raw applicant columns and returns P(default).

Packing feature engineering, preprocessing and the model into one Pipeline means:
- everything learned from data (medians, scaling, categories) is fit on train only, so no leakage;
- one saved file contains everything the API needs to reproduce training exactly.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_risk.config import MODEL_INPUTS, RANDOM_SEED, TARGET
from credit_risk.features import ENGINEERED_FEATURES, feature_builder

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


# --- Gradient boosting -----------------------------------------------------------

# Column order after the feature step: raw inputs, then engineered features.
LGBM_FEATURE_ORDER = [*MODEL_INPUTS, *ENGINEERED_FEATURES]

# Monotone constraints: a higher external credit score may never *raise* predicted risk.
# This costs almost no accuracy, keeps explanations intuitive, and is the kind of
# guarantee credit regulators and model-risk teams ask for.
RISK_DECREASES_WITH = {"ext_source_1", "ext_source_2", "ext_source_3"}


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Turns text columns into pandas 'category' columns, which LightGBM handles natively.

    The list of categories is learned in fit (training data only) and reused in transform,
    so a category's internal code is the same at training and prediction time.
    Categories never seen in training become missing values.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object = None) -> "CategoryCaster":
        self.categories_ = {col: sorted(X[col].dropna().unique()) for col in self.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, categories in self.categories_.items():
            known = X[col].where(X[col].isin(categories))  # unseen category -> missing
            X[col] = pd.Categorical(known, categories=categories)
        return X


def build_lgbm_pipeline() -> Pipeline:
    """Gradient-boosted trees. No imputation or scaling needed: trees handle missing
    values natively (they learn which side of each split missing values go) and only
    care about the order of values, not their scale."""
    model = lgb.LGBMClassifier(
        n_estimators=3000,  # upper bound; early stopping picks the real number
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=200,  # each leaf needs 200+ applicants: guards against overfitting
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        monotone_constraints=[-1 if f in RISK_DECREASES_WITH else 0 for f in LGBM_FEATURE_ORDER],
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    return Pipeline(
        [
            ("features", feature_builder()),
            ("categories", CategoryCaster(CATEGORICAL_COLUMNS)),
            ("model", model),
        ]
    )


def fit_lgbm(train: pd.DataFrame, val: pd.DataFrame) -> Pipeline:
    """Fit with early stopping: add trees until validation log loss stops improving for
    200 rounds. Log loss (not AUC) because it rewards honest probabilities, not just a good
    ranking, and the approval cut-off relies on the probabilities being honest.

    The validation set must pass through the same preprocessing as the training set,
    so we fit the preprocessing steps first, then the model with a transformed eval set.
    """
    pipeline = build_lgbm_pipeline()
    preprocess, model = pipeline[:-1], pipeline[-1]  # the slice shares the same step objects

    X_train = preprocess.fit_transform(train[MODEL_INPUTS])
    X_val = preprocess.transform(val[MODEL_INPUTS])
    if list(X_train.columns) != LGBM_FEATURE_ORDER:
        raise ValueError("Feature order changed; monotone constraints would hit the wrong columns.")

    model.fit(
        X_train,
        train[TARGET],
        eval_X=X_val,
        eval_y=val[TARGET],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    return pipeline
