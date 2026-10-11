"""Glassbox API.

Run: uvicorn api.main:app --reload

The trained pipeline is loaded once at startup. Requests only run predictions:
nothing is ever trained at request time.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from api.schemas import (
    HealthResponse,
    ModelInfo,
    Policy,
    PredictRequest,
    PredictResponse,
    RiskSurface,
)
from credit_risk.artifacts import artifacts_dir, load_artifacts
from credit_risk.config import DEFAULT_LGD, DEFAULT_MARGIN, MODEL_INPUTS
from credit_risk.decision import cost_based_threshold, decide
from credit_risk.whatif import risk_surface

DEFAULT_POLICY = Policy(
    lgd=DEFAULT_LGD,
    margin=DEFAULT_MARGIN,
    threshold=cost_based_threshold(DEFAULT_LGD, DEFAULT_MARGIN),
)
SURFACE_FEATURES = ("ext_source_2", "ext_source_3")  # the two most informative inputs


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fails at startup (not on the first request) if the model hasn't been trained.
    app.state.pipeline, app.state.metadata = load_artifacts(artifacts_dir())
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
    )


@app.get("/model/surface", response_model=RiskSurface)
def model_surface() -> RiskSurface:
    return app.state.surface


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    # One-row DataFrame with the same columns, in the same order, as in training.
    row = pd.DataFrame([request.applicant.model_dump()])[MODEL_INPUTS]
    p_default = float(app.state.pipeline.predict_proba(row)[0, 1])
    return PredictResponse(
        probability_of_default=p_default,
        decision=decide(p_default, DEFAULT_POLICY.threshold),
        threshold=DEFAULT_POLICY.threshold,
        model_version=app.state.metadata["model_version"],
    )
