"""Central place for paths, column names and constants.

Everything else imports from here, so a column rename or path change happens once.
"""

from pathlib import Path

# src/credit_risk/config.py -> parents[2] is the repo root (works with `pip install -e .`).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_CSV = RAW_DATA_DIR / "application_train.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42
VAL_SIZE = 0.2   # fractions of the full dataset: 60 / 20 / 20
TEST_SIZE = 0.2

# In the raw data, DAYS_EMPLOYED = 365243 (~1000 years) is a placeholder used for
# pensioners and unemployed applicants, not a real value.
DAYS_EMPLOYED_SENTINEL = 365243
DAYS_PER_YEAR = 365.25

# --- Column names ---------------------------------------------------------------
# Raw Kaggle names are renamed to readable snake_case names when loading.
# After that, the training data has exactly the same columns as an API request,
# so the saved pipeline sees identical inputs at training and serving time.

ID_COL = "applicant_id"
TARGET = "defaulted"

# Raw Kaggle column -> applicant field the model may use.
MODEL_INPUT_RENAMES: dict[str, str] = {
    # Loan
    "AMT_CREDIT": "amt_credit",
    "AMT_ANNUITY": "amt_annuity",
    "AMT_GOODS_PRICE": "amt_goods_price",
    "NAME_CONTRACT_TYPE": "contract_type",
    # Income and employment
    "AMT_INCOME_TOTAL": "amt_income_total",
    "NAME_INCOME_TYPE": "income_type",
    "NAME_EDUCATION_TYPE": "education_type",
    "DAYS_EMPLOYED": "years_employed",  # converted from negative days in data.clean
    # External credit scores (0-1, bureau-like, often missing)
    "EXT_SOURCE_1": "ext_source_1",
    "EXT_SOURCE_2": "ext_source_2",
    "EXT_SOURCE_3": "ext_source_3",
    # Household and region
    "FLAG_OWN_CAR": "own_car",      # "Y"/"N" -> bool
    "FLAG_OWN_REALTY": "own_realty",  # "Y"/"N" -> bool
    "CNT_CHILDREN": "cnt_children",
    "REGION_RATING_CLIENT": "region_rating",
}

# Protected attributes: kept ONLY for the fairness audit, never fed to the model.
AUDIT_RENAMES: dict[str, str] = {
    "CODE_GENDER": "gender",
    "DAYS_BIRTH": "age_years",  # converted from negative days in data.clean
    "NAME_FAMILY_STATUS": "family_status",
}

OTHER_RENAMES: dict[str, str] = {"SK_ID_CURR": ID_COL, "TARGET": TARGET}

ALL_RENAMES = {**OTHER_RENAMES, **MODEL_INPUT_RENAMES, **AUDIT_RENAMES}
MODEL_INPUTS = list(MODEL_INPUT_RENAMES.values())
AUDIT_COLUMNS = list(AUDIT_RENAMES.values())
