import numpy as np
import pytest

from config.settings import Settings
from src.components.model_artifact import QuantileModelBundle
from src.components.model_config import MIN_TRAINING_ROWS, TARGETS
from src.pipeline.training_pipeline import (
    TrainingError,
    load_training_frame,
    train_quantile_models,
    train_target,
)
from tests.conftest import synthetic_frame

FAST_GRID = [{"num_leaves": 7, "min_child_samples": 5}]


@pytest.fixture
def fast_training(monkeypatch):
    """Use the synthetic frame instead of the database, with a one-config 2-fold CV."""
    monkeypatch.setattr(
        "src.pipeline.training_pipeline.load_training_frame",
        lambda app_settings: synthetic_frame(n_rows=160),
    )
    monkeypatch.setattr("src.components.model_config.PARAM_GRID", FAST_GRID)
    monkeypatch.setattr("src.components.model_config.N_SPLITS", 2)


def test_load_training_frame_requires_database_url():
    # Arrange
    app_settings = Settings(database_url=None)

    # Act / Assert
    with pytest.raises(TrainingError):
        load_training_frame(app_settings)


def test_train_target_rejects_too_few_labeled_rows():
    # Arrange
    features = np.zeros((MIN_TRAINING_ROWS + 10, 3))
    values = np.full(MIN_TRAINING_ROWS + 10, np.nan)
    values[:5] = 1.0

    # Act / Assert
    with pytest.raises(TrainingError):
        train_target(features, values, "hdl")


def test_train_quantile_models_saves_a_loadable_bundle(fast_training, tmp_path):
    # Arrange
    app_settings = Settings(database_url="postgresql://unused", model_dir=tmp_path)

    # Act
    bundle = train_quantile_models(app_settings)
    loaded = QuantileModelBundle.load_latest(tmp_path)

    # Assert
    assert loaded.version == bundle.version
    assert set(loaded.models) == set(TARGETS)
    assert loaded.training_row_counts == dict.fromkeys(TARGETS, 160)
    intervals = loaded.metrics["intervals"]
    assert isinstance(intervals, list)
    assert {row["target"] for row in intervals} == set(TARGETS)
