from datetime import date

from src.components.model_config import TARGETS
from src.pipeline.twin_pipeline import predict_twin
from tests.conftest import make_wearables

START = date(2026, 1, 1)


def test_twin_runs_on_production_predictor(predictor, sample_patient_codes):
    profile = {**sample_patient_codes, "height_cm": 178.0}
    labs = [{"date": START, "biomarker": "hdl", "value": 38.0}]
    twin = predict_twin(predictor, profile, labs, make_wearables(START, 90), n_samples=200)
    assert set(twin.biomarkers) == set(TARGETS)
    assert twin.prevent_note is None
    assert set(twin.prevent) == {"population_prior", "current", "forecast"}
    hdl = twin.biomarkers["hdl"].trajectory
    assert hdl["basis"].iloc[0] == "measured"
