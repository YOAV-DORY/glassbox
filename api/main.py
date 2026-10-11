"""Glassbox API.

Run: uvicorn api.main:app --reload

The trained pipeline is loaded once at startup. Requests only run predictions:
nothing is ever trained at request time.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from api.schemas import HealthResponse, PredictRequest, PredictResponse
from credit_risk.artifacts import artifacts_dir, load_artifacts
from credit_risk.config import DEFAULT_LGD, DEFAULT_MARGIN, MODEL_INPUTS
from credit_risk.decision import cost_based_threshold, decide


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fails at startup (not on the first request) if the model hasn't been trained.
    app.state.pipeline, app.state.metadata = load_artifacts(artifacts_dir())
    yield


app = FastAPI(title="Glassbox API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    metadata = app.state.metadata
    return HealthResponse(
        status="ok",
        model_version=metadata["model_version"],
        model_type=metadata["model_type"],
        validation_roc_auc=metadata["validation_metrics"]["roc_auc"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    # One-row DataFrame with the same columns, in the same order, as in training.
    row = pd.DataFrame([request.applicant.model_dump()])[MODEL_INPUTS]
    p_default = float(app.state.pipeline.predict_proba(row)[0, 1])
    threshold = cost_based_threshold(DEFAULT_LGD, DEFAULT_MARGIN)
    return PredictResponse(
        probability_of_default=p_default,
        decision=decide(p_default, threshold),
        threshold=threshold,
        model_version=app.state.metadata["model_version"],
    )
