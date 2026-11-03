"""LightGBM quantile model training and cross-validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold

from src.components import model_config
from src.components.model_config import (
    CATEGORICAL_INDICES,
    DEFAULT_N_ESTIMATORS,
    EARLY_STOPPING_ROUNDS,
    LEARNING_RATE,
    MIN_N_ESTIMATORS,
    QUANTILES,
    RANDOM_STATE,
)
from src.components.model_evaluation import FloatArray, fix_crossing, pinball_loss
from src.utils.logger import get_logger

logger = get_logger(__name__)

QuantileModels = dict[float, lgb.LGBMRegressor]
Dataset = tuple[FloatArray, FloatArray]


@dataclass(frozen=True)
class CVResult:
    """Outcome of cross-validation for one target.

    Attributes:
        config_name: Name of the best configuration.
        cfg: LightGBM parameters of the best configuration.
        n_estimators_by_tau: Mean early-stopping tree count per quantile.
        pinball_mean_by_tau: Mean validation pinball loss per quantile.
        mean_pinball: Mean validation pinball loss over all quantiles and folds.
    """

    config_name: str
    cfg: dict[str, int]
    n_estimators_by_tau: dict[float, int]
    pinball_mean_by_tau: dict[float, float]
    mean_pinball: float


def config_name(cfg: Mapping[str, int]) -> str:
    """Readable name of a hyperparameter configuration.

    Args:
        cfg: Parameters with "num_leaves" and "min_child_samples".

    Returns:
        A name such as "leaves15_mcs50".
    """
    return f"leaves{cfg['num_leaves']}_mcs{cfg['min_child_samples']}"


def make_lgbm(tau: float, params: Mapping[str, int]) -> lgb.LGBMRegressor:
    """Quantile LightGBM regressor with the project defaults.

    Args:
        tau: Quantile level in (0, 1).
        params: Overrides such as num_leaves, min_child_samples, n_estimators.

    Returns:
        An unfitted regressor.
    """
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=tau,
        n_estimators=params.get("n_estimators", DEFAULT_N_ESTIMATORS),
        learning_rate=LEARNING_RATE,
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def fit_with_early_stopping(
    model: lgb.LGBMRegressor, train: Dataset, validation: Dataset
) -> lgb.LGBMRegressor:
    """Fit with early stopping on a validation set.

    Args:
        model: Unfitted regressor.
        train: Training features and targets.
        validation: Validation features and targets.

    Returns:
        The fitted model.
    """
    model.fit(
        *train,
        eval_X=(validation[0],),
        eval_y=(validation[1],),
        eval_metric="quantile",
        categorical_feature=list(CATEGORICAL_INDICES),
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model


def labeled_rows(features: FloatArray, values: FloatArray) -> Dataset:
    """Keep the rows whose target value is present.

    Args:
        features: Feature matrix for every row.
        values: Target values, NaN where missing.

    Returns:
        Features and targets of the labeled rows.
    """
    mask = ~np.isnan(values)
    return features[mask], values[mask]


def cross_validate(
    features: FloatArray,
    values: FloatArray,
    quantiles: tuple[float, ...] = QUANTILES,
    param_grid: list[dict[str, int]] | None = None,
) -> CVResult:
    """Pick the best configuration and per-quantile tree counts with K-fold CV.

    Args:
        features: Training feature matrix.
        values: Training targets.
        quantiles: Quantile levels to fit.
        param_grid: Configurations to compare. Defaults to model_config.PARAM_GRID.

    Returns:
        The best configuration and its tree counts.
    """
    grid = param_grid or model_config.PARAM_GRID
    folds = list(
        KFold(
            n_splits=model_config.N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        ).split(features)
    )
    losses: defaultdict[tuple[int, float], list[float]] = defaultdict(list)
    best_iterations: defaultdict[tuple[int, float], list[int]] = defaultdict(list)

    for cfg_index, cfg in enumerate(grid):
        for tau in quantiles:
            for train_rows, valid_rows in folds:
                model = fit_with_early_stopping(
                    make_lgbm(tau, cfg),
                    (features[train_rows], values[train_rows]),
                    (features[valid_rows], values[valid_rows]),
                )
                prediction = np.asarray(
                    model.predict(features[valid_rows]), dtype=np.float64
                )
                losses[cfg_index, tau].append(
                    pinball_loss(values[valid_rows], prediction, tau)
                )
                best_iterations[cfg_index, tau].append(
                    int(model.best_iteration_ or model.n_estimators)
                )
        logger.info(
            "  %s: mean pinball=%.4f",
            config_name(cfg),
            np.mean([losses[cfg_index, tau] for tau in quantiles]),
        )

    mean_by_cfg = [
        float(np.mean([losses[index, tau] for tau in quantiles]))
        for index in range(len(grid))
    ]
    best = int(np.argmin(mean_by_cfg))
    return CVResult(
        config_name=config_name(grid[best]),
        cfg=grid[best],
        n_estimators_by_tau={
            tau: max(
                MIN_N_ESTIMATORS, round(float(np.mean(best_iterations[best, tau])))
            )
            for tau in quantiles
        },
        pinball_mean_by_tau={
            tau: float(np.mean(losses[best, tau])) for tau in quantiles
        },
        mean_pinball=mean_by_cfg[best],
    )


def fit_final(
    features: FloatArray,
    values: FloatArray,
    cfg: Mapping[str, int],
    n_estimators_by_tau: Mapping[float, int],
    quantiles: tuple[float, ...] = QUANTILES,
) -> QuantileModels:
    """Fit one quantile model per tau on all given rows.

    Args:
        features: Feature matrix.
        values: Targets.
        cfg: LightGBM parameters.
        n_estimators_by_tau: Tree count per quantile.
        quantiles: Quantile levels to fit.

    Returns:
        Fitted models keyed by tau.
    """
    models: QuantileModels = {}
    for tau in quantiles:
        model = make_lgbm(tau, {**cfg, "n_estimators": n_estimators_by_tau[tau]})
        model.fit(features, values, categorical_feature=list(CATEGORICAL_INDICES))
        models[tau] = model
    return models


def predict_quantiles(
    models: Mapping[float, lgb.LGBMRegressor],
    features: FloatArray,
    quantiles: tuple[float, ...] = QUANTILES,
) -> FloatArray:
    """Predict a non-crossing quantile matrix.

    Args:
        models: Fitted models keyed by tau.
        features: Feature matrix.
        quantiles: Quantile levels, ascending.

    Returns:
        Predictions of shape (rows, len(quantiles)), sorted within each row.
    """
    predictions = np.column_stack([models[tau].predict(features) for tau in quantiles])
    return fix_crossing(predictions.astype(np.float64))
