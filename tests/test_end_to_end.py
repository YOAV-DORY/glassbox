"""End to end: the real training protocol on synthetic applicants, served by the production server.

Unlike the API tests (which use a model fitted directly in a fixture), this runs train.run(),
the same function as `python -m credit_risk.train` and the demo, then serves what it saved
through api/serve.py, the one-port server used by the demo script and the Docker image.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from credit_risk.config import PROJECT_ROOT
from credit_risk.reports import MetricsReport
from credit_risk.synthetic import make_synthetic_raw
from credit_risk.train import run
from tests.conftest import VALID_APPLICANT


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    out_dir = tmp_path_factory.mktemp("artifacts")
    reports_dir = tmp_path_factory.mktemp("reports")
    report = run(make_synthetic_raw(n=4000, seed=1), out_dir, reports_dir, dataset="synthetic-test",
                 n_boot=50, verbose=False)
    return out_dir, reports_dir, report


@pytest.fixture(scope="module")
def web_dist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A stand-in for the built web app (the real one needs Node to build)."""
    dist = tmp_path_factory.mktemp("dist")
    (dist / "index.html").write_text('<!doctype html><div id="root"></div>')
    (dist / "favicon.svg").write_text("<svg/>")
    (dist.parent / "outside-secret.txt").write_text("SECRET")  # must never be reachable
    return dist


@pytest.fixture
def server(trained, web_dist: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from api.serve import app

    monkeypatch.setenv("GLASSBOX_ARTIFACTS_DIR", str(trained[0]))
    monkeypatch.setenv("GLASSBOX_WEB_DIST", str(web_dist))
    with TestClient(app) as client:  # runs the startup code that loads the trained model
        yield client


def test_training_writes_everything_the_api_and_readers_need(trained) -> None:
    out_dir, reports_dir, report = trained
    for name in ("model.joblib", "metadata.json", "validation_predictions.parquet"):
        assert (out_dir / name).exists(), name
    assert not (out_dir / "test_predictions.parquet").exists()  # the test set is never saved for reuse
    MetricsReport.model_validate_json((reports_dir / "metrics.json").read_text())
    fairness = json.loads((reports_dir / "fairness.json").read_text())
    assert fairness["cohort"] == "validation" and fairness["gender"] and fairness["age_band"]
    assert report["dataset"] == "synthetic-test"


def test_trained_model_serves_predictions_reasons_and_portfolio(server: TestClient) -> None:
    info = server.get("/api/model/info").json()
    assert info["dataset"] == "synthetic-test"

    body = server.post("/api/predict", json={"applicant": VALID_APPLICANT}).json()
    assert 0 < body["probability_of_default"] < 1
    if info["model_type"] == "lgbm":
        assert body["reasons_up"] or body["reasons_down"]

    portfolio = server.post("/api/portfolio/simulate", json={}).json()
    assert portfolio["cohort"] == "validation" and len(portfolio["calibration"]) == 10


def test_web_routes_and_path_traversal(server: TestClient, web_dist: Path) -> None:
    assert server.get("/favicon.svg").text == "<svg/>"  # real files are served as they are
    for path in ("/", "/assess", "/portfolio"):  # app routes fall back to index.html
        assert 'id="root"' in server.get(path).text
    # The file really exists one level up, so only the server's guard keeps it private.
    assert (web_dist / ".." / "outside-secret.txt").resolve().is_file()
    for attempt in ("/..%2Foutside-secret.txt", "/%2E%2E/outside-secret.txt"):
        response = server.get(attempt)
        assert "SECRET" not in response.text, attempt


def test_committed_metrics_report_matches_the_schema() -> None:
    """reports/metrics.json in the repo is what the README quotes; it must stay readable."""
    path = PROJECT_ROOT / "reports" / "metrics.json"
    report = MetricsReport.model_validate_json(path.read_text())
    assert report.dataset == "home-credit"
    assert report.selected_model in report.test


def test_schema_rejects_a_broken_report() -> None:
    good = json.loads((PROJECT_ROOT / "reports" / "metrics.json").read_text())
    for broken in ({k: v for k, v in good.items() if k != "test"}, {**good, "surprise": 1}):
        with pytest.raises(ValueError):
            MetricsReport.model_validate(broken)
