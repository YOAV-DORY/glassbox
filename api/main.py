"""Glassbox API.

Run: uvicorn api.main:app --reload

The trained pipeline is loaded once at startup. Requests only run predictions:
nothing is ever trained at request time.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from api.schemas import HealthResponse, ModelInfo, Policy, PredictRequest, PredictResponse
from credit_risk.artifacts import artifacts_dir, load_artifacts
from credit_risk.config import DEFAULT_LGD, DEFAULT_MARGIN, MODEL_INPUTS
from credit_risk.decision import cost_based_threshold, decide

DEFAULT_POLICY = Policy(
    lgd=DEFAULT_LGD,
    margin=DEFAULT_MARGIN,
    threshold=cost_based_threshold(DEFAULT_LGD, DEFAULT_MARGIN),
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fails at startup (not on the first request) if the model hasn't been trained.
    app.state.pipeline, app.state.metadata = load_artifacts(artifacts_dir())
    yield


app = FastAPI(title="Glassbox API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_version=app.state.metadata["model_version"])


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    metadata = app.state.metadata
    metrics = metadata["validation_metrics"]
    return ModelInfo(
        model_version=metadata["model_version"],
        model_type=metadata["model_type"],
        trained_at=metadata["trained_at"],
        validation_roc_auc=metrics["roc_auc"],
        validation_pr_auc=metrics["pr_auc"],
        average_default_rate=metrics["actual_default_rate"],
        policy=DEFAULT_POLICY,
    )


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
