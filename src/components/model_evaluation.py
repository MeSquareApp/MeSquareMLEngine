"""Evaluation helpers for quantile regression models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_pinball_loss


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, tau: float) -> float:
    return float(mean_pinball_loss(y, q_pred, alpha=tau))


def fix_crossing(pred_matrix: np.ndarray) -> np.ndarray:
    """Monotone rearrangement across quantile columns."""
    return np.sort(pred_matrix, axis=1)


def skill_score(model_loss: float, baseline_loss: float) -> float:
    if baseline_loss <= 0:
        return float("nan")
    return 1.0 - model_loss / baseline_loss


def coverage(y: np.ndarray, q_pred: np.ndarray) -> float:
    return float(np.mean(y <= q_pred))


def interval_metrics(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    cover = float(np.mean((y >= lower) & (y <= upper)))
    width = float(np.mean(upper - lower))
    return {"coverage": cover, "mean_width": width}


def evaluate_target(
    y_test: np.ndarray,
    pred_matrix: np.ndarray,
    y_train: np.ndarray,
    quantiles: tuple[float, ...],
    target: str,
    config_name: str,
    n_estimators_by_tau: dict[float, int],
) -> tuple[list[dict], dict[str, float]]:
    """Return per-quantile eval rows and the p10-p90 interval metrics."""
    eval_rows: list[dict] = []
    q_map: dict[float, np.ndarray] = {}

    for j, tau in enumerate(quantiles):
        q_pred = pred_matrix[:, j]
        q_map[tau] = q_pred
        baseline = float(np.quantile(y_train, tau))
        model_loss = pinball_loss(y_test, q_pred, tau)
        base_loss = pinball_loss(y_test, np.full_like(y_test, baseline, dtype=float), tau)
        eval_rows.append(
            {
                "target": target,
                "tau": tau,
                "config": config_name,
                "n_estimators": n_estimators_by_tau[tau],
                "test_pinball": model_loss,
                "baseline_pinball": base_loss,
                "skill": skill_score(model_loss, base_loss),
                "coverage": coverage(y_test, q_pred),
            }
        )

    interval = interval_metrics(y_test, q_map[0.10], q_map[0.90])
    interval_row = {
        "target": target,
        "interval": "80%",
        "coverage": interval["coverage"],
        "mean_width": interval["mean_width"],
    }
    return eval_rows, interval_row
