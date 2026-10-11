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
    assert body["model_version"].startswith("lgbm-")


def test_model_info_reports_metrics_and_policy(client: TestClient) -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert 0 < body["average_default_rate"] < 1
    policy = body["policy"]
    assert policy["threshold"] == pytest.approx(policy["margin"] / (policy["margin"] + policy["lgd"]))


def test_model_surface_is_a_square_grid_of_probabilities(client: TestClient) -> None:
    body = client.get("/model/surface").json()
    assert len(body["z"]) == len(body["y"]) and all(len(row) == len(body["x"]) for row in body["z"])
    assert all(0 < p < 1 for row in body["z"] for p in row)


def test_predict_returns_probability_and_consistent_decision(client: TestClient) -> None:
    response = client.post("/predict", json={"applicant": VALID_APPLICANT})
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {"probability_of_default", "decision", "threshold", "model_version",
                         "expected_profit", "reasons_up", "reasons_down"}
    assert 0 < body["probability_of_default"] < 1
    for reason in body["reasons_up"]:
        assert reason["impact"] > 0 and reason["label"] and reason["value"]
    for reason in body["reasons_down"]:
        assert reason["impact"] < 0
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


def test_predict_follows_the_policy(client: TestClient) -> None:
    strict = {"lgd": 0.9, "margin": 0.02}  # cut-off 0.02 / 0.92: about 2%
    body = client.post("/predict", json={"applicant": VALID_APPLICANT, "policy": strict}).json()
    assert body["threshold"] == pytest.approx(0.02 / 0.92)
    # Positive expected profit exactly when the probability is under the formula's cut-off.
    assert (body["expected_profit"] > 0) == (body["probability_of_default"] < body["threshold"])

    manual = client.post("/predict", json={"applicant": VALID_APPLICANT, "policy": {"threshold_override": 0.99}}).json()
    assert manual["threshold"] == 0.99 and manual["decision"] == "approve"


def test_portfolio_simulation_contract(client: TestClient) -> None:
    response = client.post("/portfolio/simulate", json={"lgd": 0.6, "margin": 0.1})
    assert response.status_code == 200
    body = response.json()
    assert body["formula_threshold"] == pytest.approx(0.1 / 0.7)
    assert body["policy"]["threshold"] == pytest.approx(0.1 / 0.7)
    assert body["approve_all"]["approval_rate"] == 1.0
    assert len(body["curve"]) == 60
    rates = [point["approval_rate"] for point in body["curve"]]
    assert rates == sorted(rates)  # a higher cut-off never approves fewer people
    assert body["cohort"] == "validation"
    assert [row["decile"] for row in body["calibration"]] == list(range(1, 11))
    assert {row["group"] for row in body["fairness"]["gender"]} <= {"F", "M"}
    assert all(0 <= row["approval_ratio"] <= 1 for row in body["fairness"]["age_band"])


@pytest.mark.parametrize("policy", [{"lgd": 0}, {"margin": 1.5}, {"threshold_override": 1}, {"risk": 1}])
def test_portfolio_rejects_invalid_policies(client: TestClient, policy: dict) -> None:
    assert client.post("/portfolio/simulate", json=policy).status_code == 422
