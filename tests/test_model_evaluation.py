import numpy as np
import pytest

from src.components.model_evaluation import (
    EvalContext,
    coverage,
    evaluate_target,
    fix_crossing,
    interval_metrics,
    pinball_loss,
    skill_score,
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


def test_coverage_and_interval_metrics():
    # Arrange
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.zeros(4)
    upper = np.full(4, 2.5)

    # Act
    below = coverage(observed, upper)
    interval = interval_metrics(observed, lower, upper)

    # Assert
    assert below == 0.5
    assert interval == {"coverage": 0.5, "mean_width": 2.5}


def test_evaluate_target_returns_one_row_per_quantile_and_interval():
    # Arrange
    rng = np.random.default_rng(0)
    y_train = rng.normal(size=200)
    y_test = rng.normal(size=100)
    predictions = np.tile(np.quantile(y_train, QUANTILES), (100, 1))
    context = EvalContext("hdl", "leaves15_mcs50", dict.fromkeys(QUANTILES, 60))

    # Act
    rows, interval = evaluate_target(y_test, predictions, y_train, QUANTILES, context)

    # Assert
    assert [row["tau"] for row in rows] == list(QUANTILES)
    assert all(row["skill"] == pytest.approx(0.0) for row in rows)
    assert interval["target"] == "hdl"
    assert interval["interval"] == "80%"
