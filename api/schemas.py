"""Request and response shapes for the API.

Field names match the model's input columns exactly (see credit_risk.config), so a
validated request can go straight into the pipeline. Pydantic rejects bad input with a
422 error that names the field, before it ever reaches the model.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class PredictRequest(BaseModel):
    applicant: Applicant
    # M4 adds an optional lending policy (LGD, margin) here.


class PredictResponse(BaseModel):
    probability_of_default: float
    decision: Literal["approve", "decline"]
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    model_type: str
    validation_roc_auc: float
