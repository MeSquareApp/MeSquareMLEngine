from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.components.personalization.biomarker_config import BIOMARKERS, DRIVER_FEATURES
from src.components.personalization.distributions import measurement_variance
from src.components.personalization.kalman import (
    Prior,
    confidence_label,
    effect_path,
    run_filter,
)

START = date(2026, 1, 1)


def zero_deltas(n_days: int) -> pl.DataFrame:
    days = pl.date_range(
        START, START + timedelta(days=n_days - 1), interval="1d", eager=True
    )
    return (
        days.alias("date")
        .to_frame()
        .with_columns(pl.lit(0.0).alias(name) for name in DRIVER_FEATURES)
    )


def test_no_labs_no_wearables_keeps_prior_mean():
    # Arrange
    spec = BIOMARKERS["non_hdl"]
    deltas = zero_deltas(30)

    # Act
    result = run_filter(
        spec, Prior(150.0, 40.0), [], deltas, START + timedelta(days=29)
    )

    # Assert
    trajectory = result.trajectory
    np.testing.assert_allclose(trajectory["mean"].to_numpy(), 150.0)
    assert set(trajectory["basis"]) == {"prior"}
    assert trajectory["sd"].is_sorted()


def test_single_lab_matches_conjugate_update():
    # Arrange
    spec = BIOMARKERS["hdl"]
    prior, lab = Prior(50.0, 14.0), 38.0
    noise = measurement_variance(lab, spec.measurement_cv, spec.transform)
    expected_var = 1.0 / (1.0 / prior.sd**2 + 1.0 / noise)
    expected_mean = expected_var * (prior.mean / prior.sd**2 + lab / noise)

    # Act
    result = run_filter(spec, prior, [(START, lab)], zero_deltas(1), START)

    # Assert
    row = result.trajectory.row(0, named=True)
    assert row["mean"] == pytest.approx(expected_mean)
    assert row["sd"] == pytest.approx(np.sqrt(expected_var))
    assert row["basis"] == "measured"


def test_lab_shrinks_uncertainty_and_it_regrows():
    # Arrange
    spec = BIOMARKERS["hba1c"]

    # Act
    result = run_filter(
        spec, Prior(np.log(5.5), 0.1), [(START, 6.1)], zero_deltas(365), START
    )

    # Assert
    sd = result.trajectory["sd"]
    assert sd[0] < 0.1 * 0.35
    assert sd[-1] > sd[0]
    assert set(result.trajectory["basis"][1:]) == {"anchored"}


def test_wearable_effect_moves_mean_with_lag():
    # Arrange
    spec = BIOMARKERS["sbp"]
    deltas = (
        zero_deltas(200)
        .with_row_index()
        .with_columns(
            pl.when(pl.col("index") >= 30).then(4.0).otherwise(0.0).alias("steps")
        )
        .drop("index")
    )
    target = spec.effects["steps"] * 4.0

    # Act
    result = run_filter(
        spec, Prior(130.0, 15.0), [], deltas, START + timedelta(days=199)
    )

    # Assert
    effect = result.trajectory["effect"]
    assert effect[29] == 0
    assert 0 > effect[40] > target
    assert effect[-1] == pytest.approx(target, rel=0.01)
    assert result.trajectory["basis"][-1] == "estimated"


def test_effect_path_matches_recurrence():
    # Arrange
    spec = BIOMARKERS["hdl"]
    deltas = zero_deltas(10).with_columns(pl.lit(2.0).alias("steps"))
    rate = 1.0 - np.exp(-1.0 / spec.tau_days)
    expected = [0.0]
    for _ in range(9):
        expected.append(
            expected[-1] + (spec.effects["steps"] * 2.0 - expected[-1]) * rate
        )

    # Act
    effect = effect_path(spec, deltas)

    # Assert
    np.testing.assert_allclose(effect, expected)


def test_innovation_z_is_zero_when_lab_matches_prediction():
    # Arrange
    spec = BIOMARKERS["non_hdl"]

    # Act
    result = run_filter(
        spec, Prior(150.0, 40.0), [(START, 150.0)], zero_deltas(1), START
    )

    # Assert
    assert result.innovations[0].z == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("sd", "label"), [(0.1, "high"), (0.5, "medium"), (0.9, "low")]
)
def test_confidence_label(sd, label):
    # Arrange
    prior_sd = 1.0

    # Act
    result = confidence_label(sd, prior_sd)

    # Assert
    assert result == label
