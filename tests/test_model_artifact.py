import orjson
import pytest

from src.components.model_artifact import (
    METADATA_FILENAME,
    ArtifactError,
    QuantileModelBundle,
    new_version_id,
)
from src.components.model_config import FEATURES, QUANTILES, TARGETS


def test_save_and_load_round_trip(tiny_bundle, tmp_path):
    # Arrange
    tiny_bundle.version = "20260101T0000Z"
    tiny_bundle.interval_widening = {"hdl": {"all": 1.5, "female:40-59": 2.0}}

    # Act
    tiny_bundle.save(tmp_path)
    loaded = QuantileModelBundle.load(tmp_path / "20260101T0000Z")

    # Assert
    assert loaded.version == "20260101T0000Z"
    assert loaded.targets == TARGETS
    assert loaded.quantiles == QUANTILES
    assert loaded.feature_list == list(FEATURES)
    assert set(loaded.models["hdl"]) == set(QUANTILES)
    assert loaded.interval_widening == {"hdl": {"all": 1.5, "female:40-59": 2.0}}


def test_save_writes_readable_metadata(tiny_bundle, tmp_path):
    # Arrange
    tiny_bundle.version = "20260101T0000Z"

    # Act
    folder = tiny_bundle.save(tmp_path)
    metadata = orjson.loads((folder / METADATA_FILENAME).read_bytes())

    # Assert
    assert metadata["version"] == "20260101T0000Z"
    assert metadata["targets"] == list(TARGETS)
    assert "models" not in metadata


def test_load_latest_picks_newest_version(tiny_bundle, tmp_path):
    # Arrange
    root = tmp_path / "models"
    for version in ("20260101T0000Z", "20260301T0000Z", "20260201T0000Z"):
        tiny_bundle.version = version
        tiny_bundle.save(root)

    # Act
    latest = QuantileModelBundle.load_latest(root)

    # Assert
    assert latest.version == "20260301T0000Z"


@pytest.mark.parametrize("make_dir", [False, True])
def test_load_latest_raises_without_bundles(tmp_path, make_dir):
    # Arrange
    root = tmp_path / "models"
    if make_dir:
        root.mkdir()

    # Act / Assert
    with pytest.raises(ArtifactError):
        QuantileModelBundle.load_latest(root)


def test_new_version_id_format():
    # Arrange
    expected_length = len("20261103T1542Z")

    # Act
    version = new_version_id()

    # Assert
    assert len(version) == expected_length
    assert version[8] == "T"
    assert version.endswith("Z")
