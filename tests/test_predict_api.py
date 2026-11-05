from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app import app
from src.components.model_config import TARGETS
from src.pipeline.twin_pipeline import TwinEngine
from tests.conftest import make_registry


@pytest.fixture
def client(monkeypatch, predictor, mock_prevent) -> Iterator[TestClient]:
    """App with the tiny trained predictor and mocked PREVENT."""
    engine = TwinEngine(
        prior=predictor, registry=make_registry(), prevent_version="test"
    )
    monkeypatch.setattr("app._load_engine", lambda: engine)
    monkeypatch.setattr("app.prevent_available", lambda: True)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_model_version(client, predictor):
    # Arrange
    expected = predictor.version

    # Act
    body = client.get("/health").json()

    # Assert
    assert body == {"status": "healthy", "model_version": expected, "prevent": "test"}


def test_predict_returns_population_bands_and_estimated_risk(client, sample_patient):
    # Arrange
    request = {"patients": [{**sample_patient, "height_cm": 178.0}]}

    # Act
    response = client.post("/v1/predict", json=request)

    # Assert
    assert response.status_code == 200
    body = response.json()
    result = body["results"][0]
    assert {p["biomarker"] for p in result["predictions"]} == set(TARGETS)
    assert {p["basis"] for p in result["predictions"]} == {"population"}
    assert all(p["p10"] <= p["p50"] <= p["p90"] for p in result["predictions"])
    assert result["risk"]["status"] == "generated"
    assert result["risk"]["risk_basis"] == "estimated"
    first = result["risk"]["results"][0]
    assert first["risk_low"] <= first["risk_pct"] <= first["risk_high"]
    assert body["model_versions"]["prior"] == "testbundle"
    assert result["input_hash"].startswith("sha256:")


def test_predict_without_height_still_returns_bands(client, sample_patient):
    # Arrange
    request = {"patients": [sample_patient]}

    # Act
    result = client.post("/v1/predict", json=request).json()["results"][0]

    # Assert
    assert len(result["predictions"]) == len(TARGETS)
    assert result["risk"]["status"] == "not_generated"
    assert result["risk"]["missing_inputs"] == ["height_cm"]


def test_predict_rejects_unknown_ethnicity(client, sample_patient):
    # Arrange
    request = {"patients": [{**sample_patient, "race_eth": "Martian"}]}

    # Act
    response = client.post("/v1/predict", json=request)

    # Assert
    assert response.status_code == 422


def test_predict_returns_503_without_model(monkeypatch, sample_patient):
    # Arrange
    monkeypatch.setattr("app._load_engine", lambda: None)
    monkeypatch.setattr("app.prevent_available", lambda: True)

    # Act
    with TestClient(app) as test_client:
        response = test_client.post("/v1/predict", json={"patients": [sample_patient]})

    # Assert
    assert response.status_code == 503
