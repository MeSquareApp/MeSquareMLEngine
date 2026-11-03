from datetime import date, timedelta

import numpy as np
import pytest

from src.components.egfr import egfr_ckd_epi_2021
from src.components.model_config import TARGETS
from src.pipeline.twin_pipeline import (
    LabRecord,
    TwinInputs,
    biomarker_samples,
    normalize_labs,
    predict_twin,
)
from tests.conftest import STUB_PRIOR, make_wearables

START = date(2026, 1, 1)


def test_normalize_labs_converts_total_chol_and_creatinine(twin_profile_codes):
    # Arrange
    labs: list[LabRecord] = [
        {"date": START, "biomarker": "total_chol", "value": 230.0},
        {"date": START, "biomarker": "hdl", "value": 38.0},
        {"date": START, "biomarker": "scr", "value": 1.0},
        {"date": START + timedelta(days=1), "biomarker": "total_chol", "value": 210.0},
    ]

    # Act
    result = normalize_labs(labs, twin_profile_codes)

    # Assert
    assert result["non_hdl"] == [(START, 192.0)]
    assert result["hdl"] == [(START, 38.0)]
    [(egfr_day, egfr)] = result["egfr"]
    assert egfr_day == START
    assert egfr == pytest.approx(egfr_ckd_epi_2021(1.0, 52, 1))


def test_normalize_labs_prefers_direct_values_over_derived(twin_profile_codes):
    # Arrange: one blood draw reported both directly and as inputs to a conversion.
    labs: list[LabRecord] = [
        {"date": START, "biomarker": "total_chol", "value": 230.0},
        {"date": START, "biomarker": "hdl", "value": 38.0},
        {"date": START, "biomarker": "non_hdl", "value": 192.0},
        {"date": START, "biomarker": "scr", "value": 1.0},
        {"date": START, "biomarker": "egfr", "value": 85.0},
    ]

    # Act
    result = normalize_labs(labs, twin_profile_codes)

    # Assert
    assert result["non_hdl"] == [(START, 192.0)]
    assert result["egfr"] == [(START, 85.0)]


def test_normalize_labs_derives_once_per_day(twin_profile_codes):
    # Arrange
    labs: list[LabRecord] = [
        {"date": START, "biomarker": "hdl", "value": 40.0},
        {"date": START, "biomarker": "total_chol", "value": 200.0},
        {"date": START, "biomarker": "total_chol", "value": 200.0},
    ]

    # Act
    result = normalize_labs(labs, twin_profile_codes)

    # Assert
    assert result["non_hdl"] == [(START, 160.0)]


def test_twin_runs_on_production_predictor(
    predictor, sample_patient_codes, mock_prevent
):
    # Arrange
    inputs = TwinInputs(
        profile={**sample_patient_codes, "height_cm": 178.0},
        labs=[{"date": START, "biomarker": "hdl", "value": 38.0}],
        wearable_days=make_wearables(START, 90),
    )

    # Act
    twin = predict_twin(predictor, inputs, n_samples=200)

    # Assert
    assert set(twin.biomarkers) == set(TARGETS)
    assert twin.prevent_note is None
    assert set(twin.prevent) == {"population_prior", "current", "forecast"}
    assert twin.biomarkers["hdl"].trajectory["basis"][0] == "measured"


def test_twin_without_height_explains_missing_prevent(
    stub_prior, twin_profile_codes, mock_prevent
):
    # Arrange
    inputs = TwinInputs(profile={**twin_profile_codes, "height_cm": None}, as_of=START)

    # Act
    twin = predict_twin(stub_prior, inputs, n_samples=50)

    # Assert
    assert twin.prevent == {}
    assert twin.prevent_note == "height_cm is required for BMI"


def test_twin_reports_unavailable_r_package(
    stub_prior, twin_profile_codes, monkeypatch
):
    # Arrange
    monkeypatch.setattr("src.pipeline.twin_pipeline.prevent_available", lambda: False)
    inputs = TwinInputs(profile=twin_profile_codes, as_of=START)

    # Act
    twin = predict_twin(stub_prior, inputs, n_samples=50)

    # Assert
    assert twin.prevent == {}
    assert twin.prevent_note == "AHA PREVENT R package is not available"


def test_biomarker_samples_from_prior_have_prior_median(
    stub_prior, twin_profile_codes, mock_prevent
):
    # Arrange
    twin = predict_twin(stub_prior, TwinInputs(profile=twin_profile_codes, as_of=START))

    # Act
    samples = biomarker_samples(twin, None, n_samples=4000, seed=1)

    # Assert
    assert np.median(samples["hdl"]) == pytest.approx(STUB_PRIOR["hdl"][0], rel=0.02)


def test_biomarker_samples_reject_dates_outside_timeline(
    stub_prior, twin_profile_codes, mock_prevent
):
    # Arrange
    twin = predict_twin(
        stub_prior, TwinInputs(profile=twin_profile_codes, as_of=START, horizon_days=0)
    )

    # Act / Assert
    with pytest.raises(ValueError):
        biomarker_samples(twin, START + timedelta(days=5), n_samples=10, seed=0)
