from datetime import date

import pytest
from fastapi.testclient import TestClient

from app import app
from tests.conftest import StubPrior, make_wearables

START = date(2026, 1, 1)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        # Lifespan loads whatever model is on disk. Swap in the stub after startup.
        app.state.predictor = StubPrior()
        yield test_client


def payload(profile: dict, labs: list[dict], n_days: int) -> dict:
    days = make_wearables(START, n_days)
    days["date"] = days["date"].astype(str)
    return {
        "profile": profile,
        "labs": labs,
        "wearable_days": days.to_dict(orient="records"),
        "horizon_days": 60,
    }


DAY0_LABS = [
    {"date": str(START), "biomarker": "total_chol", "value": 230},
    {"date": str(START), "biomarker": "hdl", "value": 38},
    {"date": str(START), "biomarker": "hba1c", "value": 5.9},
    {"date": str(START), "biomarker": "scr", "value": 1.0},
    {"date": str(START), "biomarker": "sbp", "value": 138},
]


def _width(biomarker: dict) -> float:
    return biomarker["current"]["q90"] - biomarker["current"]["q10"]


def test_scenario_a_wearables_only(client, twin_profile):
    body = client.post("/v1/twin/predict", json=payload(twin_profile, [], 120)).json()
    sbp = body["biomarkers"]["sbp"]
    assert body["wearable"]["baseline_ready"]
    assert sbp["current"]["basis"] == "estimated"
    assert sbp["current"]["q50"] < sbp["prior"]["q50"]
    assert len(sbp["trajectory"]) == 120 + 60
    assert set(body["prevent"]) == {"population_prior", "current", "forecast"}


def test_scenario_a_before_baseline_returns_prior(client, twin_profile):
    body = client.post("/v1/twin/predict", json=payload(twin_profile, [], 20)).json()
    hdl = body["biomarkers"]["hdl"]
    assert not body["wearable"]["baseline_ready"]
    assert hdl["current"]["basis"] == "prior"
    assert hdl["current"]["q50"] == pytest.approx(hdl["prior"]["q50"])


def test_scenario_b_lab_at_signup(client, twin_profile):
    with_lab = client.post("/v1/twin/predict", json=payload(twin_profile, DAY0_LABS, 120)).json()
    without = client.post("/v1/twin/predict", json=payload(twin_profile, [], 120)).json()
    hdl = with_lab["biomarkers"]["hdl"]
    assert hdl["current"]["basis"] == "anchored"
    assert abs(hdl["current"]["q50"] - 38) < abs(hdl["prior"]["q50"] - 38)
    without_hdl = without["biomarkers"]["hdl"]
    assert _width(hdl) < _width(without_hdl)
    assert len(hdl["innovations"]) == 1
    assert with_lab["biomarkers"]["non_hdl"]["innovations"][0]["observed"] == 192


def test_missing_model_returns_503(twin_profile):
    with TestClient(app) as test_client:
        app.state.predictor = None
        response = test_client.post("/v1/twin/predict", json={"profile": twin_profile})
    assert response.status_code == 503
