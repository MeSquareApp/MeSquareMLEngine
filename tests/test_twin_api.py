from collections.abc import Iterator
from datetime import date

import pytest
from fastapi.testclient import TestClient

from app import app
from tests.conftest import StubPrior, make_wearables

START = date(2026, 1, 1)
HORIZON_DAYS = 60
DAY0_LABS = [
    {"date": str(START), "biomarker": "total_chol", "value": 230},
    {"date": str(START), "biomarker": "hdl", "value": 38},
    {"date": str(START), "biomarker": "hba1c", "value": 5.9},
    {"date": str(START), "biomarker": "scr", "value": 1.0},
    {"date": str(START), "biomarker": "sbp", "value": 138},
]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, mock_prevent: None) -> Iterator[TestClient]:
    """App with a stub prior and mocked PREVENT. Nothing is read from disk or R."""
    monkeypatch.setattr("app._load_predictor", StubPrior)
    monkeypatch.setattr("app.prevent_available", lambda: True)
    with TestClient(app) as test_client:
        yield test_client


def payload(profile: dict, labs: list[dict], n_days: int) -> dict:
    days = [
        {**day, "date": day["date"].isoformat()}  # type: ignore[union-attr]
        for day in make_wearables(START, n_days)
    ]
    return {
        "profile": profile,
        "labs": labs,
        "wearable_days": days,
        "horizon_days": HORIZON_DAYS,
    }


def interval_width(biomarker: dict) -> float:
    return biomarker["current"]["q90"] - biomarker["current"]["q10"]


def test_scenario_a_wearables_only(client, twin_profile):
    # Arrange
    request = payload(twin_profile, [], 120)

    # Act
    body = client.post("/v1/twin/predict", json=request).json()

    # Assert
    sbp = body["biomarkers"]["sbp"]
    assert body["wearable"]["baseline_ready"]
    assert sbp["current"]["basis"] == "estimated"
    assert sbp["current"]["q50"] < sbp["prior"]["q50"]
    assert len(sbp["trajectory"]) == 120 + HORIZON_DAYS
    assert set(body["prevent"]) == {"population_prior", "current", "forecast"}
    assert isinstance(body["prevent_out_of_range"], dict)


def test_scenario_a_before_baseline_returns_prior(client, twin_profile):
    # Arrange
    request = payload(twin_profile, [], 20)

    # Act
    body = client.post("/v1/twin/predict", json=request).json()

    # Assert
    hdl = body["biomarkers"]["hdl"]
    assert not body["wearable"]["baseline_ready"]
    assert hdl["current"]["basis"] == "prior"
    assert hdl["current"]["q50"] == pytest.approx(hdl["prior"]["q50"])


def test_scenario_b_lab_at_signup(client, twin_profile):
    # Arrange
    with_lab_request = payload(twin_profile, DAY0_LABS, 120)
    without_lab_request = payload(twin_profile, [], 120)

    # Act
    with_lab = client.post("/v1/twin/predict", json=with_lab_request).json()
    without_lab = client.post("/v1/twin/predict", json=without_lab_request).json()

    # Assert
    hdl = with_lab["biomarkers"]["hdl"]
    assert hdl["current"]["basis"] == "anchored"
    assert abs(hdl["current"]["q50"] - 38) < abs(hdl["prior"]["q50"] - 38)
    assert interval_width(hdl) < interval_width(without_lab["biomarkers"]["hdl"])
    assert len(hdl["innovations"]) == 1
    assert with_lab["biomarkers"]["non_hdl"]["innovations"][0]["observed"] == 192


def test_missing_model_returns_503(monkeypatch, twin_profile):
    # Arrange
    monkeypatch.setattr("app._load_predictor", lambda: None)
    monkeypatch.setattr("app.prevent_available", lambda: True)

    # Act
    with TestClient(app) as test_client:
        response = test_client.post("/v1/twin/predict", json={"profile": twin_profile})

    # Assert
    assert response.status_code == 503
