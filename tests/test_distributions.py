import numpy as np
import pytest

from src.components.personalization.distributions import (
    Z_10_90,
    daily_process_variance,
    forward,
    inverse,
    measurement_variance,
    natural_quantiles,
    normal_from_quantiles,
)


@pytest.mark.parametrize("transform", ["identity", "log"])
def test_forward_and_inverse_round_trip(transform):
    # Arrange
    values = np.array([0.5, 5.4, 120.0])

    # Act
    round_trip = inverse(forward(values, transform), transform)

    # Assert
    np.testing.assert_allclose(round_trip, values)


def test_normal_from_quantiles_identity():
    # Arrange
    quantiles = {0.10: 40.0, 0.50: 50.0, 0.90: 60.0}

    # Act
    mean, sd = normal_from_quantiles(quantiles, "identity")

    # Assert
    assert mean == 50.0
    assert sd == pytest.approx(20.0 / Z_10_90)


def test_normal_from_quantiles_log_uses_log_scale():
    # Arrange
    quantiles = {0.10: 5.0, 0.50: 5.5, 0.90: 6.2}

    # Act
    mean, sd = normal_from_quantiles(quantiles, "log")

    # Assert
    assert mean == pytest.approx(np.log(5.5))
    assert sd == pytest.approx((np.log(6.2) - np.log(5.0)) / Z_10_90)


def test_natural_quantiles_recover_inputs():
    # Arrange
    quantiles = {0.10: 40.0, 0.50: 50.0, 0.90: 60.0}
    mean, sd = normal_from_quantiles(quantiles, "identity")

    # Act
    result = natural_quantiles(np.array([mean]), np.array([sd]), "identity")

    # Assert
    assert result["q10"][0] == pytest.approx(40.0)
    assert result["q50"][0] == pytest.approx(50.0)
    assert result["q90"][0] == pytest.approx(60.0)


def test_measurement_variance_scales_with_value_on_identity():
    # Arrange
    cv = 0.07

    # Act
    small, large = (measurement_variance(v, cv, "identity") for v in (40.0, 80.0))

    # Assert
    assert large == pytest.approx(4 * small)
    assert measurement_variance(40.0, cv, "log") == pytest.approx(np.log1p(cv**2))


def test_daily_process_variance_adds_up_to_annual_cv():
    # Arrange
    level, annual_cv = 100.0, 0.08

    # Act
    yearly = 365 * daily_process_variance(level, annual_cv, "identity")

    # Assert
    assert yearly == pytest.approx((annual_cv * level) ** 2)
