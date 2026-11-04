"""Unit tests for the PREVENT wrapper. The R call is mocked."""

import numpy as np
import pytest

from src.components.prevent import (
    SEX_FEMALE,
    SEX_MALE,
    PreventInputs,
    clip_to_validated,
    prevent_risk,
)
from tests.conftest import fake_prevent_base


def make_inputs(**overrides: object) -> PreventInputs:
    values: dict[str, object] = {
        "age": 50,
        "sex": SEX_FEMALE,
        "total_chol": 200,
        "hdl": 45,
        "sbp": 160,
        "bmi": 35,
        "egfr": 90,
        "diabetes": 1,
        "smoking": 0,
        "bp_treated": 1,
        "statin": 0,
    }
    values.update(overrides)
    return PreventInputs(**values)  # type: ignore[arg-type]


def test_prevent_risk_converts_percent_to_fraction(mock_prevent):
    # Arrange
    inputs = make_inputs()
    expected = (
        fake_prevent_base(
            np.array([1.0]),
            np.array([50.0]),
            {"sbp": np.array([160.0]), "tc": np.array([200.0])},
        )["prevent_base_10yr_CVD"][0]
        / 100
    )

    # Act
    risk = prevent_risk(inputs)

    # Assert
    assert risk["total_cvd", 10][0] == pytest.approx(expected)
    assert set(risk) == {
        (outcome, horizon)
        for outcome in ("total_cvd", "ascvd", "heart_failure")
        for horizon in (10, 30)
    }


def test_prevent_risk_maps_sex_to_official_codes(monkeypatch):
    # Arrange
    seen: list[np.ndarray] = []

    def capture(sex, age, columns):
        seen.append(sex)
        return fake_prevent_base(sex, age, columns)

    monkeypatch.setattr("src.components.prevent._call_prevent_base", capture)

    # Act
    prevent_risk(make_inputs(sex=SEX_MALE))
    prevent_risk(make_inputs(sex=SEX_FEMALE))

    # Assert
    assert seen[0][0] == 0.0
    assert seen[1][0] == 1.0


def test_prevent_risk_broadcasts_scalars_to_sample_length(mock_prevent):
    # Arrange
    inputs = make_inputs(sbp=np.array([120.0, 140.0, 160.0]))

    # Act
    risk = prevent_risk(inputs)

    # Assert
    assert risk["total_cvd", 10].shape == (3,)
    assert np.all(np.diff(risk["total_cvd", 10]) > 0)


@pytest.mark.parametrize("overrides", [{"age": 25}, {"age": 80}, {"sex": 3}])
def test_prevent_risk_rejects_invalid_inputs(mock_prevent, overrides):
    # Arrange
    inputs = make_inputs(**overrides)

    # Act / Assert
    with pytest.raises(ValueError):
        prevent_risk(inputs)


def test_clip_to_validated():
    # Arrange
    values = np.array([10.0, 50.0, 150.0])

    # Act
    clipped = clip_to_validated("hdl", values)

    # Assert
    np.testing.assert_allclose(clipped, [20.0, 50.0, 100.0])


def test_thirty_year_risk_only_up_to_age_59(mock_prevent):
    # Arrange
    young, old = make_inputs(age=50), make_inputs(age=65)

    # Act
    young_risk, old_risk = prevent_risk(young), prevent_risk(old)

    # Assert
    assert not np.isnan(young_risk["total_cvd", 30][0])
    assert np.isnan(old_risk["total_cvd", 30][0])
    assert not np.isnan(old_risk["total_cvd", 10][0])
