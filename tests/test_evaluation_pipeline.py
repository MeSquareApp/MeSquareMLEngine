from config.settings import Settings
from src.components.personalization.activation import load_registry
from src.pipeline.evaluation_pipeline import evaluate_lab_stage, held_out_rows
from tests.conftest import synthetic_frame


def test_held_out_rows_are_a_cluster_split():
    # Arrange
    frame = synthetic_frame(n_rows=400)

    # Act
    rows = held_out_rows(frame, "hdl")

    # Assert
    assert 0 < rows.height < frame.height / 2
    assert (rows["wt_lab"] > 0).all()


def test_evaluate_lab_stage_writes_a_valid_registry(monkeypatch, tiny_bundle, tmp_path):
    # Arrange
    frame = synthetic_frame(n_rows=400)
    monkeypatch.setattr(
        "src.pipeline.evaluation_pipeline.load_training_frame",
        lambda app_settings: frame,
    )
    monkeypatch.setattr(
        "src.pipeline.evaluation_pipeline.QuantileModelBundle.load_latest",
        lambda root: tiny_bundle,
    )
    path = tmp_path / "activation.json"

    # Act
    registry = evaluate_lab_stage(Settings(database_url="postgresql://unused"), path)

    # Assert
    loaded = load_registry(path)
    assert loaded == registry
    assert set(loaded.lab) == {"non_hdl", "hdl", "hba1c", "egfr", "sbp"}
    assert not any(decision.active for decision in loaded.wearable.values())
    assert loaded.lab["hdl"].evidence["method"] == "simulation"


def test_holdout_predictions_are_calibrated_bands(tiny_bundle):
    # Arrange
    from src.pipeline.evaluation_pipeline import grouped_split, holdout_predictions

    tiny_bundle.metrics["cv"] = {
        "hdl": {"cfg": {"num_leaves": 7, "min_child_samples": 5},
                "n_estimators_by_tau": {"0.1": 20, "0.5": 20, "0.9": 20}}
    }  # fmt: skip
    train, test = grouped_split(synthetic_frame(n_rows=400), "hdl")

    # Act
    bands = holdout_predictions(tiny_bundle, train, test, "hdl")

    # Assert
    assert bands.shape == (test.height, 3)
    assert (bands[:, 0] <= bands[:, 1]).all() and (bands[:, 1] <= bands[:, 2]).all()
