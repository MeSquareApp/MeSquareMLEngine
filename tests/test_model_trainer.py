import numpy as np
import pytest

from src.components.feature_engineering import prepare_features
from src.components.model_config import MIN_N_ESTIMATORS, QUANTILES
from src.components.model_trainer import (
    config_name,
    cross_validate,
    fit_final,
    labeled_rows,
    make_lgbm,
    predict_quantiles,
)
from tests.conftest import TINY_CONFIG, synthetic_frame

SMALL_GRID = [
    {"num_leaves": 7, "min_child_samples": 5},
    {"num_leaves": 15, "min_child_samples": 20},
]


@pytest.fixture
def training_data():
    frame = synthetic_frame(n_rows=150)
    return prepare_features(frame), frame["hdl"].to_numpy()


def test_config_name():
    # Arrange
    cfg = {"num_leaves": 15, "min_child_samples": 50}

    # Act
    name = config_name(cfg)

    # Assert
    assert name == "leaves15_mcs50"


def test_make_lgbm_sets_quantile_objective():
    # Arrange
    tau = 0.9

    # Act
    model = make_lgbm(tau, {**TINY_CONFIG, "n_estimators": 10})

    # Assert
    params = model.get_params()
    assert (params["objective"], params["alpha"], params["n_estimators"]) == (
        "quantile",
        0.9,
        10,
    )


def test_labeled_rows_drops_missing_targets():
    # Arrange
    features = np.arange(6, dtype=float).reshape(3, 2)
    values = np.array([1.0, np.nan, 3.0])

    # Act
    kept_features, kept_values = labeled_rows(features, values)

    # Assert
    np.testing.assert_array_equal(kept_features, [[0.0, 1.0], [4.0, 5.0]])
    np.testing.assert_array_equal(kept_values, [1.0, 3.0])


def test_cross_validate_picks_a_grid_config(training_data, monkeypatch):
    # Arrange
    monkeypatch.setattr("src.components.model_config.N_SPLITS", 2)
    features, values = training_data

    # Act
    result = cross_validate(features, values, QUANTILES, SMALL_GRID)

    # Assert
    assert result.cfg in SMALL_GRID
    assert result.config_name == config_name(result.cfg)
    assert set(result.n_estimators_by_tau) == set(QUANTILES)
    assert min(result.n_estimators_by_tau.values()) >= MIN_N_ESTIMATORS
    assert result.mean_pinball > 0


def test_fit_final_and_predict_quantiles_are_ordered(training_data):
    # Arrange
    features, values = training_data
    trees = dict.fromkeys(QUANTILES, 30)

    # Act
    models = fit_final(features, values, TINY_CONFIG, trees, QUANTILES)
    predictions = predict_quantiles(models, features[:10], QUANTILES)

    # Assert
    assert set(models) == set(QUANTILES)
    assert predictions.shape == (10, len(QUANTILES))
    assert np.all(np.diff(predictions, axis=1) >= 0)
