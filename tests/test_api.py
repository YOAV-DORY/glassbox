"""API contract tests: response shapes, and that bad input is rejected with 422."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.conftest import VALID_APPLICANT


@pytest.fixture
def client(trained_artifacts_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("GLASSBOX_ARTIFACTS_DIR", str(trained_artifacts_dir))
    with TestClient(app) as client:  # "with" runs the startup code that loads the model
        yield client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith("logreg-")


def test_predict_returns_probability_and_consistent_decision(client: TestClient) -> None:
    response = client.post("/predict", json={"applicant": VALID_APPLICANT})
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {"probability_of_default", "decision", "threshold", "model_version"}
    assert 0 < body["probability_of_default"] < 1
    expected = "approve" if body["probability_of_default"] < body["threshold"] else "decline"
    assert body["decision"] == expected


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("amt_income_total", -5),        # amounts must be positive
        ("ext_source_2", 1.5),           # scores are between 0 and 1
        ("income_type", "Astronaut"),    # unknown category
        ("region_rating", 4),            # only 1, 2, 3 exist
    ],
)
def test_predict_rejects_invalid_values(client: TestClient, field: str, bad_value: object) -> None:
    response = client.post("/predict", json={"applicant": {**VALID_APPLICANT, field: bad_value}})
    assert response.status_code == 422
    assert field in str(response.json()["detail"])


def test_predict_rejects_unknown_and_missing_fields(client: TestClient) -> None:
    typo = {**VALID_APPLICANT, "amt_credt": 1}
    assert client.post("/predict", json={"applicant": typo}).status_code == 422

    missing = {k: v for k, v in VALID_APPLICANT.items() if k != "amt_credit"}
    assert client.post("/predict", json={"applicant": missing}).status_code == 422
