"""Glassbox API.

Run: uvicorn api.main:app --reload

The trained pipeline is loaded once at startup. Requests only run predictions:
nothing is ever trained at request time.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    CurvePoint,
    FeatureImportance,
    HealthResponse,
    ModelInfo,
    Policy,
    PolicyRequest,
    PortfolioOutcome,
    PortfolioResponse,
    PredictRequest,
    PredictResponse,
    Reason,
    RiskSurface,
)
from credit_risk.artifacts import artifacts_dir, load_artifacts, load_cohort
from credit_risk.config import DEFAULT_LGD, DEFAULT_MARGIN, MODEL_INPUTS
from credit_risk.decision import cost_based_threshold, decide, expected_profit, portfolio_outcome, profit_curve
from credit_risk.explain import Explainer
from credit_risk.whatif import risk_surface

DEFAULT_POLICY = Policy(
    lgd=DEFAULT_LGD,
    margin=DEFAULT_MARGIN,
    threshold=cost_based_threshold(DEFAULT_LGD, DEFAULT_MARGIN),
)
SURFACE_FEATURES = ("ext_source_2", "ext_source_3")  # the two most informative inputs
CURVE_THRESHOLDS = np.round(np.arange(0.01, 0.601, 0.01), 2)  # cut-offs from 1% to 60%


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fails at startup (not on the first request) if the model hasn't been trained.
    app.state.pipeline, app.state.metadata = load_artifacts(artifacts_dir())
    # SHAP's TreeExplainer only works for tree models; a logistic regression gets no reasons.
    is_tree = app.state.metadata["model_type"] == "lgbm"
    app.state.explainer = Explainer(app.state.pipeline) if is_tree else None
    app.state.portfolio = load_cohort(artifacts_dir())  # test-set predictions + outcomes
    # The surface never changes for a given model, so compute it once and reuse it.
    x, y, z = risk_surface(app.state.pipeline, app.state.metadata["reference_applicant"], *SURFACE_FEATURES)
    app.state.surface = RiskSurface(
        x_feature=SURFACE_FEATURES[0], y_feature=SURFACE_FEATURES[1], x=x, y=y, z=z,
        threshold=DEFAULT_POLICY.threshold,
    )
    yield


app = FastAPI(title="Glassbox API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_version=app.state.metadata["model_version"])


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    metadata = app.state.metadata
    val, test = metadata["validation_metrics"], metadata.get("test_metrics") or {}
    return ModelInfo(
        model_version=metadata["model_version"],
        model_type=metadata["model_type"],
        trained_at=metadata["trained_at"],
        train_rows=metadata["train_rows"],
        validation_roc_auc=val["roc_auc"],
        validation_pr_auc=val["pr_auc"],
        test_roc_auc=test.get("roc_auc"),
        test_pr_auc=test.get("pr_auc"),
        average_default_rate=val["actual_default_rate"],
        policy=DEFAULT_POLICY,
        global_importance=[FeatureImportance(**item) for item in metadata.get("global_importance", [])],
    )


@app.get("/model/surface", response_model=RiskSurface)
def model_surface() -> RiskSurface:
    return app.state.surface


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    # One-row DataFrame with the same columns, in the same order, as in training.
    row = pd.DataFrame([request.applicant.model_dump()])[MODEL_INPUTS]
    p_default = float(app.state.pipeline.predict_proba(row)[0, 1])
    up, down = app.state.explainer.reasons(row) if app.state.explainer else ([], [])
    policy = request.policy
    threshold = policy.threshold()
    return PredictResponse(
        probability_of_default=p_default,
        decision=decide(p_default, threshold),
        threshold=threshold,
        model_version=app.state.metadata["model_version"],
        expected_profit=expected_profit(p_default, request.applicant.amt_credit, policy.lgd, policy.margin),
        reasons_up=[Reason(**vars(r)) for r in up],
        reasons_down=[Reason(**vars(r)) for r in down],
    )


@app.post("/portfolio/simulate", response_model=PortfolioResponse)
def simulate_portfolio(policy: PolicyRequest) -> PortfolioResponse:
    """Replay a lending policy on the test applicants, whose real outcomes we know."""
    data = app.state.portfolio
    if data is None:
        raise HTTPException(503, "No test-set predictions yet. Run: python -m credit_risk.train")
    p, y, amount = data["p_default"].to_numpy(), data["defaulted"].to_numpy(), data["amt_credit"].to_numpy()

    def outcome(threshold: float) -> PortfolioOutcome:
        return PortfolioOutcome(**portfolio_outcome(p, y, amount, threshold, policy.lgd, policy.margin))

    curve = profit_curve(p, y, amount, policy.lgd, policy.margin, CURVE_THRESHOLDS)
    best = max(curve, key=lambda point: point["profit"])
    return PortfolioResponse(
        applicants=len(p),
        formula_threshold=cost_based_threshold(policy.lgd, policy.margin),
        best_threshold=best["threshold"],
        policy=outcome(policy.threshold()),
        approve_all=outcome(1.0 + 1e-9),  # every probability is below this, so everyone is approved
        cutoff_50=outcome(0.5),
        curve=[CurvePoint(**{k: point[k] for k in CurvePoint.model_fields}) for point in curve],
    )
