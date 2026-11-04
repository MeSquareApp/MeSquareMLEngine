import pytest

from src.components.model_config import QUANTILES, TARGET_UNITS, TARGETS


def test_predict_returns_labelled_quantiles_per_target(predictor, sample_patient_codes):
    # Arrange
    records = [sample_patient_codes, {**sample_patient_codes, "age": 70.0}]

    # Act
    results = predictor.predict(records)

    # Assert
    assert len(results) == 2
    for result in results:
        assert set(result["predictions"]) == set(TARGETS)
        for target, prediction in result["predictions"].items():
            assert prediction["unit"] == TARGET_UNITS[target]
            assert prediction["p10"] <= prediction["p50"] <= prediction["p90"]


def test_predict_quantiles_matches_batch_prediction(predictor, sample_patient_codes):
    # Arrange
    batch = predictor.predict([sample_patient_codes])[0]["predictions"]

    # Act
    single = predictor.predict_quantiles(sample_patient_codes)

    # Assert
    assert set(single) == set(TARGETS)
    assert list(single["hdl"]) == list(QUANTILES)
    assert single["hdl"][0.5] == batch["hdl"]["p50"]


def test_predict_applies_conformal_widening(predictor, sample_patient_codes):
    # Arrange
    before = predictor.predict_quantiles(sample_patient_codes)["hdl"]
    predictor.bundle.interval_widening["hdl"] = {"all": 5.0}

    # Act
    after = predictor.predict_quantiles(sample_patient_codes)["hdl"]

    # Assert
    assert after[0.1] == pytest.approx(before[0.1] - 5.0)
    assert after[0.5] == pytest.approx(before[0.5])
    assert after[0.9] == pytest.approx(before[0.9] + 5.0)


def test_version_comes_from_bundle(predictor):
    # Arrange
    expected = predictor.bundle.version

    # Act
    version = predictor.version

    # Assert
    assert version == expected
