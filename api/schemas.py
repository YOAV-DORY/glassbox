"""Request and response shapes for the API.

Field names match the model's input columns exactly (see credit_risk.config), so a
validated request can go straight into the pipeline. Pydantic rejects bad input with a
422 error that names the field, before it ever reaches the model.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from credit_risk.config import DEFAULT_LGD, DEFAULT_MARGIN
from credit_risk.decision import cost_based_threshold

ContractType = Literal["Cash loans", "Revolving loans"]
IncomeType = Literal[
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Unemployed",
    "Student",
    "Businessman",
    "Maternity leave",
]
EducationType = Literal[
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]

Amount = Field(gt=0, le=1e9)
Score = Field(default=None, ge=0, le=1, description="External credit score; null if unavailable")


class Applicant(BaseModel):
    # Reject unknown fields: a typo like "amt_credt" should fail loudly, not be ignored.
    model_config = ConfigDict(extra="forbid")

    amt_credit: float = Amount
    amt_annuity: float = Amount
    amt_goods_price: float | None = Field(default=None, gt=0, le=1e9)
    contract_type: ContractType
    amt_income_total: float = Amount
    income_type: IncomeType
    education_type: EducationType
    years_employed: float | None = Field(default=None, ge=0, le=60, description="null = not employed")
    ext_source_1: float | None = Score
    ext_source_2: float | None = Score
    ext_source_3: float | None = Score
    own_car: bool
    own_realty: bool
    cnt_children: int = Field(ge=0, le=20)
    region_rating: Literal[1, 2, 3]


class PolicyRequest(BaseModel):
    """Lending economics chosen by the user. Every field has a default, so it can be omitted."""

    model_config = ConfigDict(extra="forbid")

    lgd: float = Field(DEFAULT_LGD, gt=0, le=1, description="Share of the loan lost on a default")
    margin: float = Field(DEFAULT_MARGIN, gt=0, le=1, description="Profit on a repaid loan, as a share of it")
    threshold_override: float | None = Field(None, gt=0, lt=1, description="Set the cut-off by hand")

    def threshold(self) -> float:
        if self.threshold_override is not None:
            return self.threshold_override
        return cost_based_threshold(self.lgd, self.margin)


class PredictRequest(BaseModel):
    applicant: Applicant
    policy: PolicyRequest = Field(default_factory=PolicyRequest)


class Reason(BaseModel):
    """One factor behind a prediction (a SHAP contribution)."""

    feature: str
    label: str  # plain-language name, e.g. "Credit bureau score 2"
    value: str  # the applicant's value, formatted, e.g. "0.48" or "no score on file"
    impact: float  # contribution in log-odds; positive raises risk, negative lowers it


class PredictResponse(BaseModel):
    probability_of_default: float
    decision: Literal["approve", "decline"]
    threshold: float
    model_version: str
    expected_profit: float  # of approving this loan, under the policy, before knowing the outcome
    # Strongest factors first. Reason-code-style explanations, not a compliant adverse-action notice.
    reasons_up: list[Reason]
    reasons_down: list[Reason]


class FeatureImportance(BaseModel):
    feature: str
    label: str
    importance: float  # mean |SHAP| over 10,000 test applicants, in log-odds


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str


class Policy(BaseModel):
    lgd: float
    margin: float
    threshold: float


class ModelInfo(BaseModel):
    model_version: str
    model_type: str
    trained_at: str
    train_rows: int
    validation_roc_auc: float
    validation_pr_auc: float
    test_roc_auc: float | None  # held-out test set, scored once after model selection
    test_pr_auc: float | None
    average_default_rate: float  # in the validation data; context for a single prediction
    policy: Policy
    global_importance: list[FeatureImportance]  # most influential features first


class RiskSurface(BaseModel):
    """Predicted default risk over a grid of two inputs, all else held at a typical applicant."""

    x_feature: str
    y_feature: str
    x: list[float]
    y: list[float]
    z: list[list[float]]  # z[i][j] = probability of default at y[i], x[j]
    threshold: float


class PortfolioOutcome(BaseModel):
    """What a cut-off would have done to the validation applicants, using their real outcomes."""

    threshold: float
    approval_rate: float
    bad_rate: float  # share of approved loans that defaulted
    approved: int
    expected_loss: float  # from the model's probabilities alone
    realized_loss: float  # from what actually happened
    expected_profit: float
    realized_profit: float


class CurvePoint(BaseModel):
    threshold: float
    approval_rate: float
    bad_rate: float
    expected_profit: float
    realized_profit: float


class CalibrationBin(BaseModel):
    decile: int  # 1 = the 10% of applicants with the lowest predicted risk
    n: int
    low: float  # smallest and largest prediction in the decile
    high: float
    mean_predicted: float
    actual_rate: float
    actual_se: float  # standard error of actual_rate


class FairnessGroup(BaseModel):
    group: str
    n: int
    default_rate: float
    default_rate_se: float
    mean_predicted: float
    approval_rate: float
    default_rate_approved: float | None  # None when nobody in the group is approved
    repayers_declined: float
    approval_ratio: float  # approval rate / highest group's approval rate; < 0.8 is a warning sign


class FairnessView(BaseModel):
    gender: list[FairnessGroup]
    age_band: list[FairnessGroup]
    proxy: dict[str, float]  # how strongly the "not employed" flag stands in for age


class PortfolioResponse(BaseModel):
    cohort: Literal["validation"]  # the test set is never used for interactive analysis
    applicants: int
    formula_threshold: float  # margin / (margin + lgd)
    best_threshold: float  # most profitable cut-off on this cohort; near the formula if calibrated
    policy: PortfolioOutcome  # the chosen cut-off (formula, or the manual override)
    approve_all: PortfolioOutcome
    cutoff_50: PortfolioOutcome  # the naive 50% cut-off
    curve: list[CurvePoint]
    calibration: list[CalibrationBin]
    fairness: FairnessView  # at the chosen cut-off
