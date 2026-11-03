import numpy as np
import pytest

from src.components.model_evaluation import (
    EvalContext,
    conformal_widening,
    coverage,
    evaluate_target,
    fix_crossing,
    interval_metrics,
    pinball_loss,
    skill_score,
    weighted_quantile,
    widen_interval,
)

QUANTILES = (0.10, 0.50, 0.90)


def test_pinball_loss_matches_definition():
    # Arrange
    observed = np.array([1.0, 3.0])
    predicted = np.array([2.0, 2.0])
    tau = 0.9

    # Act
    loss = pinball_loss(observed, predicted, tau)

    # Assert
    assert loss == pytest.approx(((1 - tau) * 1.0 + tau * 1.0) / 2)


def test_pinball_loss_uses_weights():
    # Arrange
    observed = np.array([1.0, 3.0])
    predicted = np.array([2.0, 2.0])
    weights = np.array([3.0, 1.0])

    # Act
    loss = pinball_loss(observed, predicted, 0.9, weights)

    # Assert
    assert loss == pytest.approx((3 * 0.1 + 1 * 0.9) / 4)


def test_weighted_quantile_follows_the_weights():
    # Arrange
    values = np.array([1.0, 2.0, 3.0])

    # Act
    unweighted = weighted_quantile(values, np.ones(3), 0.5)
    weighted = weighted_quantile(values, np.array([1.0, 1.0, 10.0]), 0.5)

    # Assert
    assert unweighted == 2.0
    assert weighted == 3.0


def test_fix_crossing_sorts_each_row():
    # Arrange
    predictions = np.array([[3.0, 1.0, 2.0]])

    # Act
    fixed = fix_crossing(predictions)

    # Assert
    np.testing.assert_array_equal(fixed, [[1.0, 2.0, 3.0]])


@pytest.mark.parametrize(
    ("model_loss", "baseline_loss", "expected"),
    [(1.0, 2.0, 0.5), (2.0, 2.0, 0.0)],
)
def test_skill_score(model_loss, baseline_loss, expected):
    # Arrange
    losses = (model_loss, baseline_loss)

    # Act
    score = skill_score(*losses)

    # Assert
    assert score == pytest.approx(expected)


def test_skill_score_is_nan_without_positive_baseline():
    # Arrange
    baseline_loss = 0.0

    # Act
    score = skill_score(1.0, baseline_loss)

    # Assert
    assert np.isnan(score)


def test_weighted_coverage_and_interval_metrics():
    # Arrange
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.zeros(4)
    upper = np.full(4, 2.5)
    weights = np.array([3.0, 1.0, 1.0, 1.0])

    # Act
    below = coverage(observed, upper, weights)
    interval = interval_metrics(observed, lower, upper, weights)

    # Assert
    assert below == pytest.approx(4 / 6)
    assert interval == pytest.approx({"coverage": 4 / 6, "mean_width": 2.5})


def test_evaluate_target_returns_one_row_per_quantile_and_interval():
    # Arrange
    rng = np.random.default_rng(0)
    y_train = rng.normal(size=200)
    y_test = rng.normal(size=100)
    train_weights = np.ones(200)
    baseline = [weighted_quantile(y_train, train_weights, tau) for tau in QUANTILES]
    predictions = np.tile(baseline, (100, 1))
    context = EvalContext("hdl", "leaves15_mcs50", dict.fromkeys(QUANTILES, 60))

    # Act
    rows, interval = evaluate_target(
        (y_test, np.ones(100)),
        predictions,
        (y_train, train_weights),
        QUANTILES,
        context,
    )

    # Assert
    assert [row["tau"] for row in rows] == list(QUANTILES)
    assert all(row["skill"] == pytest.approx(0.0) for row in rows)
    assert interval["target"] == "hdl"
    assert interval["interval"] == "80%"


def test_weighted_quantile_handles_level_one():
    # Arrange
    values = np.array([1.0, 2.0, 3.0])

    # Act
    top = weighted_quantile(values, np.ones(3), 1.0)

    # Assert
    assert top == 3.0


def test_conformal_widening_restores_target_coverage():
    # Arrange: true 80% interval is about +-1.28, the model predicts +-0.5.
    rng = np.random.default_rng(1)
    calibration, fresh = rng.normal(size=4000), rng.normal(size=4000)
    narrow = np.full(4000, 0.5)
    weights = np.ones(4000)

    # Act
    widening = conformal_widening(calibration, -narrow, narrow, weights)
    widened = widen_interval(
        np.column_stack([-narrow, np.zeros(4000), narrow]), QUANTILES, widening
    )
    covered = coverage(fresh, widened[:, 2], weights) - coverage(
        fresh, widened[:, 0], weights
    )

    # Assert
    assert widening == pytest.approx(1.28 - 0.5, abs=0.06)
    assert covered == pytest.approx(0.8, abs=0.02)


def test_widen_interval_moves_only_the_bounds_and_keeps_order():
    # Arrange
    predictions = np.array([[1.0, 2.0, 3.0]])

    # Act
    widened = widen_interval(predictions, QUANTILES, 0.5)
    narrowed = widen_interval(predictions, QUANTILES, -2.0)

    # Assert
    np.testing.assert_array_equal(widened, [[0.5, 2.0, 3.5]])
    assert np.all(np.diff(narrowed, axis=1) >= 0)
