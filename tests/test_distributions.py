import math

import pytest

from src.components.personalization.distributions import (
    Z_90,
    band,
    drift_sigma,
    log_sigma,
    normal_from_quantiles,
)


def test_log_sigma():
    # Arrange
    cv = 0.07

    # Act
    sigma = log_sigma(cv)

    # Assert
    assert sigma == pytest.approx(math.sqrt(math.log1p(cv**2)))


def test_drift_grows_with_square_root_of_time():
    # Arrange
    annual = 0.08

    # Act
    one_year, four_years, past = (drift_sigma(annual, days) for days in (365, 1460, -5))

    # Assert
    assert four_years == pytest.approx(2 * one_year)
    assert past == 0.0


def test_normal_from_quantiles_round_trips_through_band():
    # Arrange
    quantiles = {0.10: 40.0, 0.50: 50.0, 0.90: 62.5}

    # Act
    mean, sd = normal_from_quantiles(quantiles)
    low, mid, high = band(math.exp(mean), sd)

    # Assert
    assert mid == pytest.approx(50.0)
    assert low * high == pytest.approx(40.0 * 62.5)
    assert sd == pytest.approx((math.log(62.5) - math.log(40.0)) / (2 * Z_90))
