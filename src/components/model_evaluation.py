"""Evaluation helpers for quantile regression models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import mean_pinball_loss

FloatArray = npt.NDArray[np.float64]
MetricRow = dict[str, str | float | int]
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


def pinball_loss(y: FloatArray, q_pred: FloatArray, tau: float) -> float:
    """Mean pinball loss of one quantile prediction.

    Args:
        y: Observed values.
        q_pred: Predicted tau-quantiles, same length as y.
        tau: Quantile level in (0, 1).

    Returns:
        Mean pinball loss. Lower is better.
    """
    return float(mean_pinball_loss(y, q_pred, alpha=tau))


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


def coverage(y: FloatArray, q_pred: FloatArray) -> float:
    """Share of observations at or below the predicted quantile.

    Args:
        y: Observed values.
        q_pred: Predicted quantiles, same length as y.

    Returns:
        Empirical coverage in [0, 1].
    """
    return float(np.mean(y <= q_pred))


def interval_metrics(
    y: FloatArray, lower: FloatArray, upper: FloatArray
) -> dict[str, float]:
    """Coverage and mean width of a prediction interval.

    Args:
        y: Observed values.
        lower: Lower interval bounds.
        upper: Upper interval bounds.

    Returns:
        {"coverage", "mean_width"}.
    """
    return {
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "mean_width": float(np.mean(upper - lower)),
    }


def evaluate_target(
    y_test: FloatArray,
    pred_matrix: FloatArray,
    y_train: FloatArray,
    quantiles: tuple[float, ...],
    context: EvalContext,
) -> tuple[list[MetricRow], MetricRow]:
    """Score one target's holdout predictions.

    Args:
        y_test: Holdout values.
        pred_matrix: Holdout predictions of shape (rows, quantiles).
        y_train: Training values, used for the constant-quantile baseline.
        quantiles: Quantile levels matching pred_matrix columns. Must include the
            INTERVAL_BOUNDS.
        context: Labels for the rows.

    Returns:
        One row per quantile, and the p10-p90 interval row.

    Raises:
        ValueError: If quantiles does not include both INTERVAL_BOUNDS.
    """
    eval_rows: list[MetricRow] = []
    for column, tau in enumerate(quantiles):
        q_pred = pred_matrix[:, column]
        baseline = np.full_like(y_test, np.quantile(y_train, tau), dtype=np.float64)
        model_loss = pinball_loss(y_test, q_pred, tau)
        base_loss = pinball_loss(y_test, baseline, tau)
        eval_rows.append(
            {
                "target": context.target,
                "tau": tau,
                "config": context.config_name,
                "n_estimators": context.n_estimators_by_tau[tau],
                "test_pinball": model_loss,
                "baseline_pinball": base_loss,
                "skill": skill_score(model_loss, base_loss),
                "coverage": coverage(y_test, q_pred),
            }
        )

    lower_tau, upper_tau = INTERVAL_BOUNDS
    lower, upper = (pred_matrix[:, quantiles.index(tau)] for tau in INTERVAL_BOUNDS)
    interval_row: MetricRow = {
        "target": context.target,
        "interval": f"{round((upper_tau - lower_tau) * 100)}%",
        **interval_metrics(y_test, lower, upper),
    }
    return eval_rows, interval_row
