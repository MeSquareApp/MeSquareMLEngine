import numpy as np
import polars as pl

from src.components.feature_engineering import (
    infer_category_levels,
    prepare_features,
    records_to_frame,
)
from src.components.model_config import DEFAULT_CATEGORY_LEVELS, FEATURES


def column(name: str) -> int:
    return FEATURES.index(name)


def test_infer_category_levels_appends_unseen_levels_after_defaults():
    # Arrange
    frame = pl.DataFrame(
        {"sex": [1.0, 2.0], "race_eth": [3.0, 9.0], "smoking": ["never", "unknown"]}
    )

    # Act
    levels = infer_category_levels(frame)

    # Assert
    assert levels["sex"] == DEFAULT_CATEGORY_LEVELS["sex"]
    assert levels["race_eth"] == [*DEFAULT_CATEGORY_LEVELS["race_eth"], 9.0]
    assert levels["smoking"] == [*DEFAULT_CATEGORY_LEVELS["smoking"], "unknown"]


def test_infer_category_levels_uses_defaults_for_missing_columns():
    # Arrange
    frame = pl.DataFrame({"age": [50.0]})

    # Act
    levels = infer_category_levels(frame)

    # Assert
    assert levels == DEFAULT_CATEGORY_LEVELS


def test_records_to_frame_ignores_unknown_keys_and_fills_missing():
    # Arrange
    records = [{"sex": 1, "age": 50, "height_cm": 180.0}]

    # Act
    frame = records_to_frame(records)

    # Assert
    assert frame.columns == list(FEATURES)
    assert frame["age"].to_list() == [50.0]
    assert frame["weight_kg"].to_list() == [None]


def test_prepare_features_encodes_categories_and_diabetes():
    # Arrange
    frame = records_to_frame(
        [
            {"sex": 2, "race_eth": 4, "smoking": "current", "diabetes_dx": " Yes "},
            {"sex": 1, "race_eth": 5, "smoking": "pipe", "diabetes_dx": "borderline"},
        ]
    )

    # Act
    features = prepare_features(frame)

    # Assert
    np.testing.assert_array_equal(features[:, column("sex")], [1.0, 0.0])
    np.testing.assert_array_equal(features[:, column("race_eth")], [3.0, np.nan])
    np.testing.assert_array_equal(features[:, column("smoking")], [2.0, np.nan])
    np.testing.assert_array_equal(features[:, column("diabetes_dx")], [1.0, 0.5])


def test_prepare_features_treats_absent_columns_as_missing():
    # Arrange
    frame = pl.DataFrame({"age": [61.0]})

    # Act
    features = prepare_features(frame)

    # Assert
    assert features.shape == (1, len(FEATURES))
    assert features[0, column("age")] == 61.0
    assert np.isnan(features[0, column("weight_kg")])
