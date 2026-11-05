from collections.abc import Iterator
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app import app
from src.pipeline.twin_pipeline import TwinEngine
from tests.conftest import START, full_panel, lab_record, wearable_records

REQUIRED_PREDICTION_FIELDS = {
    "biomarker",
    "basis",
    "p10",
    "p50",
    "p90",
    "unit",
    "as_of",
    "source_lab_id",
}


class RecordingStore:
    def __init__(self) -> None:
        self.batches: list = []

    def write(self, batch) -> int:
        self.batches.append(batch)
        return len(batch.predictions)


@pytest.fixture
def client(monkeypatch, engine: TwinEngine, mock_prevent) -> Iterator[TestClient]:
    """App with a stub engine and mocked PREVENT. Nothing is read from disk or R."""
    monkeypatch.setattr("app._load_engine", lambda: engine)
    monkeypatch.setattr("app.prevent_available", lambda: True)
    with TestClient(app) as test_client:
        app.state.shadow_store = RecordingStore()
        yield test_client


def test_state_response_has_every_required_field(client, twin_profile):
    # Arrange: HDL measured, the rest estimated (the doc's example).
    body = {"profile": twin_profile, "records": [lab_record("HDL", 52.0, "mg/dL")],
            "as_of": str(START)}  # fmt: skip

    # Act
    response = client.post("/v1/twin/state", json=body)

    # Assert
    assert response.status_code == 200
    data = response.json()
    hdl = next(p for p in data["predictions"] if p["biomarker"] == "hdl")
    non_hdl = next(p for p in data["predictions"] if p["biomarker"] == "non_hdl")
    assert REQUIRED_PREDICTION_FIELDS <= set(hdl)
    assert (hdl["basis"], hdl["source_lab_id"], hdl["p50"]) == (
        "lab",
        "HDL-2026-01-01",
        52.0,
    )
    assert (non_hdl["basis"], non_hdl["source_lab_id"]) == ("population", None)
    risk = data["risk"]
    assert risk["status"] == "generated"
    assert risk["risk_basis"] == "mixed"
    assert risk["inputs_used"]["hdl"] == {"value": 52.0, "source": "measured"}
    assert risk["inputs_used"]["non_hdl"]["source"] == "predicted"
    result = risk["results"][0]
    assert {
        "outcome",
        "horizon_years",
        "risk_pct",
        "risk_low",
        "risk_high",
        "category",
    } <= set(result)
    assert set(data["model_versions"]) >= {"prior", "prevent", "activation"}
    assert data["input_hash"].startswith("sha256:")


def test_state_reports_rejected_records(client, twin_profile):
    # Arrange
    records = [*full_panel(), lab_record("HDL", 1.0, "g/L", record_id="bad-unit"),
               {"id": "no-name", "value": 5.0}]  # fmt: skip

    # Act
    data = client.post(
        "/v1/twin/state", json={"profile": twin_profile, "records": records}
    ).json()

    # Assert
    rejected = {item["id"]: item["reason"] for item in data["rejected_records"]}
    assert set(rejected) == {"bad-unit", "no-name"}
    assert "unit" in rejected["bad-unit"]


def test_shadow_predictions_are_stored_not_returned(
    client, monkeypatch, twin_profile, engine
):
    # Arrange
    from tests.test_twin_pipeline import wearable_engine

    monkeypatch.setattr(app.state, "engine", wearable_engine(engine, active=False))
    after_lab = START + timedelta(days=1)
    records = [
        lab_record("HBA1C", 6.0, "%"),
        *wearable_records("resting_heart_rate", [60.0] * 70, "bpm", after_lab),
    ]
    body = {"user_id": "user-1", "profile": twin_profile, "records": records}

    # Act
    data = client.post("/v1/twin/state", json=body).json()

    # Assert
    batches = app.state.shadow_store.batches
    assert len(batches) == 1
    assert batches[0].user_id == "user-1"
    assert [item.biomarker for item in batches[0].predictions] == ["hba1c"]
    hba1c = next(p for p in data["predictions"] if p["biomarker"] == "hba1c")
    assert hba1c["basis"] == "lab"
    assert "shadow" not in data


def test_state_returns_503_without_model(monkeypatch, twin_profile):
    # Arrange
    monkeypatch.setattr("app._load_engine", lambda: None)
    monkeypatch.setattr("app.prevent_available", lambda: True)

    # Act
    with TestClient(app) as test_client:
        response = test_client.post("/v1/twin/state", json={"profile": twin_profile})

    # Assert
    assert response.status_code == 503
