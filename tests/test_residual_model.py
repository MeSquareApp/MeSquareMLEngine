import numpy as np
import pytest

from src.components.personalization.residual_model import (
    GATE_MIN_USERS,
    PairSet,
    ResidualModel,
    bootstrap_lower_bound,
    gate_residual_model,
    load_residual_models,
    mean_pinball,
    train_residual_model,
)


def synthetic_pairs(n_users: int, signal: float, seed: int = 0) -> PairSet:
    """Two pairs per user. With signal > 0, a feature moves the next lab."""
    rng = np.random.default_rng(seed)
    n_rows = 2 * n_users
    features = rng.normal(size=(n_rows, 4))
    anchor = np.full(n_rows, 100.0)
    noise = 0.03
    next_lab = anchor * np.exp(signal * features[:, 0] + rng.normal(0, noise, n_rows))
    spread = np.exp(1.2816 * np.sqrt(signal**2 + noise**2))
    last_lab_band = np.column_stack([anchor / spread, anchor, anchor * spread])
    return PairSet(
        user_ids=np.repeat(np.arange(n_users).astype(str), 2),
        features=features,
        anchor=anchor,
        next_lab=next_lab,
        baselines={"last_lab": last_lab_band, "population": last_lab_band * 1.0},
    )


def test_planted_signal_passes_the_gate():
    # Arrange
    pairs = synthetic_pairs(GATE_MIN_USERS + 50, signal=0.15)

    # Act
    result = gate_residual_model(pairs)

    # Assert
    assert result.passed, result.reasons
    assert result.metrics["improvement_vs_last_lab"] >= 0.10
    assert result.metrics["ci_lower_gain"] > 0


def test_null_signal_fails_the_gate():
    # Arrange
    pairs = synthetic_pairs(GATE_MIN_USERS + 50, signal=0.0)

    # Act
    result = gate_residual_model(pairs)

    # Assert
    assert not result.passed
    assert any("better than last_lab" in reason for reason in result.reasons)


def test_gate_refuses_too_few_users():
    # Arrange
    pairs = synthetic_pairs(20, signal=0.15)

    # Act
    result = gate_residual_model(pairs)

    # Assert
    assert not result.passed
    assert "the gate needs" in result.reasons[0]


def test_mean_pinball_and_bootstrap():
    # Arrange
    bands = np.array([[90.0, 100.0, 110.0]])
    difference = np.ones(10)
    users = np.arange(10).astype(str)

    # Act
    loss = mean_pinball(bands, np.array([100.0]))
    lower = bootstrap_lower_bound(difference, users)

    # Assert
    assert loss[0] == pytest.approx((0.1 * 10 + 0.0 + 0.1 * 10) / 3)
    assert lower == pytest.approx(1.0)


def test_save_and_load(tmp_path):
    # Arrange
    model = train_residual_model(synthetic_pairs(60, 0.15), "hba1c", "change", "r1")

    # Act
    model.save(tmp_path)
    loaded = load_residual_models(tmp_path)

    # Assert
    assert set(loaded) == {("hba1c", "change")}
    band = loaded["hba1c", "change"].predict_band(np.zeros((1, 4)), np.array([5.5]))
    assert band.shape == (1, 3)
    assert band[0, 0] <= band[0, 1] <= band[0, 2]
    assert isinstance(loaded["hba1c", "change"], ResidualModel)
