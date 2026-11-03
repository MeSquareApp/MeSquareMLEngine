"""Survey-weighted evaluation helpers for quantile regression models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import mean_pinball_loss

FloatArray = npt.NDArray[np.float64]
MetricRow = dict[str, str | float | int]
# Values and survey weights of one split.
WeightedSample = tuple[FloatArray, FloatArray]
INTERVAL_BOUNDS = (0.10, 0.90)


@dataclass(frozen=True)
class EvalContext:
    """Labels attached to one target's evaluation rows.

    Attributes:
        target: Biomarker name.
        config_name: Selected hyperparameter configuration.
        n_estimators_by_tau: Tree count per quantile.
    """

    target: str
    config_name: str
    n_estimators_by_tau: dict[float, int]


def pinball_loss(
    y: FloatArray, q_pred: FloatArray, tau: float, weights: FloatArray | None = None
) -> float:
    """Weighted mean pinball loss of one quantile prediction.

    Args:
        y: Observed values.
        q_pred: Predicted tau-quantiles, same length as y.
        tau: Quantile level in (0, 1).
        weights: Survey weights. None weighs every row equally.

    Returns:
        Mean pinball loss. Lower is better.
    """
    return float(mean_pinball_loss(y, q_pred, alpha=tau, sample_weight=weights))


def weighted_quantile(values: FloatArray, weights: FloatArray, tau: float) -> float:
    """Population tau-quantile of weighted values (inverse of the weighted CDF).

    Args:
        values: Observed values.
        weights: Survey weights, same length as values.
        tau: Quantile level in (0, 1).

    Returns:
        The smallest value whose weighted cumulative share reaches tau.
    """
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][np.searchsorted(cumulative, tau)])


def fix_crossing(pred_matrix: FloatArray) -> FloatArray:
    """Monotone rearrangement across quantile columns.

    Args:
        pred_matrix: Predictions of shape (rows, quantiles), quantiles ascending.

    Returns:
        The matrix with each row sorted ascending.
    """
    return np.sort(pred_matrix, axis=1)


def skill_score(model_loss: float, baseline_loss: float) -> float:
    """Relative improvement over a baseline loss.

    Args:
        model_loss: Loss of the model.
        baseline_loss: Loss of the baseline.

    Returns:
        1 - model_loss / baseline_loss, or NaN when the baseline loss is not positive.
    """
    if baseline_loss <= 0:
        return float("nan")
    return 1.0 - model_loss / baseline_loss


def coverage(y: FloatArray, q_pred: FloatArray, weights: FloatArray) -> float:
    """Weighted share of observations at or below the predicted quantile.

    Args:
        y: Observed values.
        q_pred: Predicted quantiles, same length as y.
        weights: Survey weights.

    Returns:
        Population coverage in [0, 1].
    """
    return float(np.average(y <= q_pred, weights=weights))


def interval_metrics(
    y: FloatArray, lower: FloatArray, upper: FloatArray, weights: FloatArray
) -> dict[str, float]:
    """Weighted coverage and mean width of a prediction interval.

    Args:
        y: Observed values.
        lower: Lower interval bounds.
        upper: Upper interval bounds.
        weights: Survey weights.

    Returns:
        {"coverage", "mean_width"}.
    """
    return {
        "coverage": float(np.average((y >= lower) & (y <= upper), weights=weights)),
        "mean_width": float(np.average(upper - lower, weights=weights)),
    }


def evaluate_target(
    test: WeightedSample,
    pred_matrix: FloatArray,
    train: WeightedSample,
    quantiles: tuple[float, ...],
    context: EvalContext,
) -> tuple[list[MetricRow], MetricRow]:
    """Score one target's holdout predictions, weighted to the US population.

    Args:
        test: Holdout values and survey weights.
        pred_matrix: Holdout predictions of shape (rows, quantiles).
        train: Training values and survey weights, for the constant-quantile
            baseline.
        quantiles: Quantile levels matching pred_matrix columns. Must include the
            INTERVAL_BOUNDS.
        context: Labels for the rows.

    Returns:
        One row per quantile, and the p10-p90 interval row.

    Raises:
        ValueError: If quantiles does not include both INTERVAL_BOUNDS.
    """
    y_test, test_weights = test
    eval_rows: list[MetricRow] = []
    for column, tau in enumerate(quantiles):
        q_pred = pred_matrix[:, column]
        baseline = np.full_like(y_test, weighted_quantile(*train, tau))
        model_loss = pinball_loss(y_test, q_pred, tau, test_weights)
        base_loss = pinball_loss(y_test, baseline, tau, test_weights)
        eval_rows.append(
            {
                "target": context.target,
                "tau": tau,
                "config": context.config_name,
                "n_estimators": context.n_estimators_by_tau[tau],
                "test_pinball": model_loss,
                "baseline_pinball": base_loss,
                "skill": skill_score(model_loss, base_loss),
                "coverage": coverage(y_test, q_pred, test_weights),
            }
        )

    lower_tau, upper_tau = INTERVAL_BOUNDS
    lower, upper = (pred_matrix[:, quantiles.index(tau)] for tau in INTERVAL_BOUNDS)
    interval_row: MetricRow = {
        "target": context.target,
        "interval": f"{round((upper_tau - lower_tau) * 100)}%",
        **interval_metrics(y_test, lower, upper, test_weights),
    }
    return eval_rows, interval_row
