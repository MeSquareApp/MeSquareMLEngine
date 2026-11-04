import numpy as np
import pytest

from src.components.calibration import (
    POOLED,
    age_band,
    calibration_groups,
    mondrian_widening,
    subgroup_coverage,
    widening_per_row,
)
from src.components.feature_engineering import prepare_features, records_to_frame

QUANTILES = (0.10, 0.50, 0.90)


@pytest.mark.parametrize(
    ("age", "band"),
    [(25.0, "20-39"), (59.5, "40-59"), (72.0, "60+"), (10.0, "unknown")],
)
def test_age_band(age, band):
    # Arrange
    years = age

    # Act
    label = age_band(years)

    # Assert
    assert label == band


def test_calibration_groups_from_feature_matrix():
    # Arrange
    features = prepare_features(
        records_to_frame([{"sex": 2, "age": 45}, {"sex": 1, "age": 70}, {"age": 30}])
    )

    # Act
    groups = calibration_groups(features)

    # Assert
    assert groups == ["female:40-59", "male:60+", "unknown:20-39"]


def test_mondrian_widening_differs_by_group_and_falls_back():
    # Arrange: group "a" is noisier than group "b"; "tiny" has too few rows.
    rng = np.random.default_rng(0)
    groups = ["a"] * 2000 + ["b"] * 2000 + ["tiny"] * 10
    noise = np.r_[rng.normal(0, 2, 2000), rng.normal(0, 0.5, 2010)]
    predictions = np.tile([-0.64, 0.0, 0.64], (4010, 1))

    # Act
    table = mondrian_widening(noise, predictions, np.ones(4010), groups, QUANTILES)
    per_row = widening_per_row(["a", "b", "tiny"], table)

    # Assert
    assert table["a"] > table["b"]
    assert "tiny" not in table
    assert per_row[2] == table[POOLED]


def test_subgroup_coverage_reports_gap_in_points():
    # Arrange
    groups = ["male:20-39", "male:20-39", "female:20-39", "female:60+"]
    covered = np.array([1.0, 1.0, 0.0, 1.0])

    # Act
    report = subgroup_coverage(covered, np.ones(4), groups)

    # Assert
    assert report["sex"]["male"] == 1.0
    assert report["sex"]["female"] == 0.5
    assert report["sex"]["gap"] == pytest.approx(50.0)
    assert report["age_band"]["gap"] == pytest.approx(100 * (1 - 2 / 3))
