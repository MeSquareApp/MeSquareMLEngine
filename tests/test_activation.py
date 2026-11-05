import pytest

from src.components.personalization.activation import (
    ActivationError,
    ActivationRegistry,
    StageDecision,
    load_registry,
    save_registry,
    validate,
)


def registry(**wearable: bool) -> ActivationRegistry:
    return ActivationRegistry(
        version="v1",
        lab={"hdl": StageDecision(True, {"error_reduction": 0.6})},
        wearable={name: StageDecision(active) for name, active in wearable.items()},
    )


def test_round_trip(tmp_path):
    # Arrange
    path = tmp_path / "activation.json"
    original = registry(hba1c=True)

    # Act
    save_registry(original, path)
    loaded = load_registry(path)

    # Assert
    assert loaded == original
    assert loaded.lab_active("hdl")
    assert loaded.wearable_active("hba1c")
    assert not loaded.wearable_active("egfr")


@pytest.mark.parametrize(
    "wearable",
    [{"egfr": True}, {"hdl": True}, {"sbp": True}],
    ids=["egfr-lab-only", "hdl-before-hba1c", "sbp-not-in-order"],
)
def test_rules_are_enforced(wearable):
    # Arrange
    candidate = registry(**wearable)

    # Act / Assert
    with pytest.raises(ActivationError):
        validate(candidate)


def test_missing_registry_raises(tmp_path):
    # Arrange
    path = tmp_path / "missing.json"

    # Act / Assert
    with pytest.raises(ActivationError):
        load_registry(path)


def test_committed_registry_is_valid():
    # Arrange / Act
    loaded = load_registry()

    # Assert
    assert not any(loaded.wearable_active(name) for name in loaded.wearable)
