import numpy as np
import pytest

from src.components.feature_engineering import prepare_features
from src.components.model_config import MIN_N_ESTIMATORS, QUANTILES
from src.components.model_trainer import (
    TrainingData,
    config_name,
    cross_validate,
    fit_final,
    labeled_rows,
    make_lgbm,
    predict_quantiles,
    select_rows,
)
from tests.conftest import TINY_CONFIG, synthetic_frame

SMALL_GRID = [
    {"num_leaves": 7, "min_child_samples": 5},
    {"num_leaves": 15, "min_child_samples": 20},
]


@pytest.fixture
def training_data() -> TrainingData:
    frame = synthetic_frame(n_rows=150)
    return TrainingData(
        prepare_features(frame),
        frame["hdl"].to_numpy(),
        frame["wt_lab"].to_numpy(),
        frame["sdmvstra"].to_numpy(),
    )


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
    # Bagging is only active when subsample_freq > 0.
    assert params["subsample"] < 1.0
    assert params["subsample_freq"] >= 1


def test_labeled_rows_drops_missing_targets_and_unusable_weights():
    # Arrange
    features = np.arange(8, dtype=float).reshape(4, 2)
    values = np.array([1.0, np.nan, 3.0, 4.0])
    weights = np.array([2.0, 2.0, 0.0, np.nan])
    clusters = np.array([1.0, 1.0, 2.0, 2.0])

    # Act
    data = labeled_rows(features, values, weights, clusters)

    # Assert
    np.testing.assert_array_equal(data.features, [[0.0, 1.0]])
    np.testing.assert_array_equal(data.values, [1.0])
    np.testing.assert_array_equal(data.weights, [2.0])
    np.testing.assert_array_equal(data.clusters, [1.0])


def test_select_rows_subsets_every_array(training_data):
    # Arrange
    rows = np.array([0, 2])

    # Act
    subset = select_rows(training_data, rows)

    # Assert
    np.testing.assert_array_equal(subset.values, training_data.values[rows])
    np.testing.assert_array_equal(subset.weights, training_data.weights[rows])
    assert subset.features.shape == (2, training_data.features.shape[1])


def test_cross_validate_picks_a_grid_config(training_data, monkeypatch):
    # Arrange
    monkeypatch.setattr("src.components.model_config.N_SPLITS", 2)

    # Act
    result = cross_validate(training_data, QUANTILES, SMALL_GRID)

    # Assert
    assert result.cfg in SMALL_GRID
    assert result.config_name == config_name(result.cfg)
    assert set(result.n_estimators_by_tau) == set(QUANTILES)
    assert min(result.n_estimators_by_tau.values()) >= MIN_N_ESTIMATORS
    assert result.mean_pinball > 0
    assert result.oof_predictions.shape == (len(training_data.values), len(QUANTILES))
    assert np.all(np.diff(result.oof_predictions, axis=1) >= 0)


def test_cross_validate_never_splits_a_cluster(training_data, monkeypatch):
    # Arrange
    monkeypatch.setattr("src.components.model_config.N_SPLITS", 2)
    seen: list[set[float]] = []
    real_fit = __import__(
        "src.components.model_trainer", fromlist=["fit_with_early_stopping"]
    ).fit_with_early_stopping

    def spy(model, train, validation):
        seen.append(set(train.clusters) & set(validation.clusters))
        return real_fit(model, train, validation)

    monkeypatch.setattr("src.components.model_trainer.fit_with_early_stopping", spy)

    # Act
    cross_validate(training_data, (0.5,), SMALL_GRID[:1])

    # Assert
    assert seen and all(not overlap for overlap in seen)


def test_fit_final_and_predict_quantiles_are_ordered(training_data):
    # Arrange
    trees = dict.fromkeys(QUANTILES, 30)

    # Act
    models = fit_final(training_data, TINY_CONFIG, trees, QUANTILES)
    predictions = predict_quantiles(models, training_data.features[:10], QUANTILES)

    # Assert
    assert set(models) == set(QUANTILES)
    assert predictions.shape == (10, len(QUANTILES))
    assert np.all(np.diff(predictions, axis=1) >= 0)


def test_survey_weights_move_the_fitted_median():
    # Arrange: identical features, two groups of outcomes. Upweighting the high
    # group should pull the population median toward it.
    features = np.zeros((200, 2))
    values = np.r_[np.full(100, 10.0), np.full(100, 20.0)]
    trees = {0.5: 50}
    clusters = np.zeros(200)
    equal = TrainingData(features, values, np.ones(200), clusters)
    skewed = TrainingData(
        features, values, np.r_[np.ones(100), np.full(100, 9.0)], clusters
    )
    cfg = {"num_leaves": 2, "min_child_samples": 1}

    # Act
    median_equal = fit_final(equal, cfg, trees, (0.5,))[0.5].predict(features[:1])
    median_skewed = fit_final(skewed, cfg, trees, (0.5,))[0.5].predict(features[:1])

    # Assert
    assert median_skewed[0] > median_equal[0]
    assert median_skewed[0] == pytest.approx(20.0, abs=0.5)
