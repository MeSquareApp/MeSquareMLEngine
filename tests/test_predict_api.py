from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app import app
from src.components.model_config import TARGETS


@pytest.fixture
def client(monkeypatch, predictor) -> Iterator[TestClient]:
    """App with the tiny predictor. Nothing is read from disk or R."""
    monkeypatch.setattr("app._load_predictor", lambda: predictor)
    monkeypatch.setattr("app.prevent_available", lambda: True)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_model_version(client, predictor):
    # Arrange
    expected = predictor.version

    # Act
    body = client.get("/health").json()

    # Assert
    assert body == {"status": "healthy", "model_version": expected}


def test_predict_accepts_nhanes_labels(client, sample_patient):
    # Arrange
    request = {"patients": [sample_patient]}

    # Act
    response = client.post("/v1/predict", json=request)

    # Assert
    assert response.status_code == 200
    predictions = response.json()["results"][0]["predictions"]
    assert set(predictions) == set(TARGETS)
    assert predictions["hdl"]["p10"] <= predictions["hdl"]["p90"]


def test_predict_rejects_unknown_ethnicity(client, sample_patient):
    # Arrange
    request = {"patients": [{**sample_patient, "race_eth": "Martian"}]}

    # Act
    response = client.post("/v1/predict", json=request)

    # Assert
    assert response.status_code == 422


def test_predict_returns_503_without_model(monkeypatch, sample_patient):
    # Arrange
    monkeypatch.setattr("app._load_predictor", lambda: None)
    monkeypatch.setattr("app.prevent_available", lambda: True)

    # Act
    with TestClient(app) as test_client:
        response = test_client.post("/v1/predict", json={"patients": [sample_patient]})

    # Assert
    assert response.status_code == 503
