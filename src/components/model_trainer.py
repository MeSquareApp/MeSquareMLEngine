"""LightGBM quantile model training and cross-validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GroupKFold

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


@dataclass(frozen=True)
class TrainingData:
    """Rows for one target, with NHANES survey weights and design clusters.

    Attributes:
        features: Feature matrix.
        values: Target values.
        weights: Survey weight per row. Larger means the row represents more people.
        clusters: NHANES design cluster (stratum x PSU) per row. Rows of one cluster
            are never split across training and validation.
    """

    features: FloatArray
    values: FloatArray
    weights: FloatArray
    clusters: FloatArray


def select_rows(data: TrainingData, rows: npt.NDArray[np.int_]) -> TrainingData:
    """Subset every array of data to the given row indices.

    Args:
        data: Rows to subset.
        rows: Row indices.

    Returns:
        The subset.
    """
    return TrainingData(
        data.features[rows], data.values[rows], data.weights[rows], data.clusters[rows]
    )


@dataclass(frozen=True)
class CVResult:
    """Outcome of cross-validation for one target.

    Attributes:
        config_name: Name of the best configuration.
        cfg: LightGBM parameters of the best configuration.
        n_estimators_by_tau: Mean early-stopping tree count per quantile.
        pinball_mean_by_tau: Mean validation pinball loss per quantile.
        mean_pinball: Mean validation pinball loss over all quantiles and folds.
        oof_predictions: Out-of-fold quantile predictions of the best configuration,
            shape (rows, quantiles), from models with the chosen tree counts.
        fold_ids: CV fold of every row.
    """

    config_name: str
    cfg: dict[str, int]
    n_estimators_by_tau: dict[float, int]
    pinball_mean_by_tau: dict[float, float]
    mean_pinball: float
    oof_predictions: FloatArray
    fold_ids: npt.NDArray[np.int_]


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
        # LightGBM only bags rows when subsample_freq > 0.
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def fit_with_early_stopping(
    model: lgb.LGBMRegressor, train: TrainingData, validation: TrainingData
) -> lgb.LGBMRegressor:
    """Fit with survey weights and weighted early stopping on a validation set.

    Args:
        model: Unfitted regressor.
        train: Training rows.
        validation: Validation rows.

    Returns:
        The fitted model.
    """
    model.fit(
        train.features,
        train.values,
        sample_weight=train.weights,
        eval_X=(validation.features,),
        eval_y=(validation.values,),
        eval_sample_weight=[validation.weights],
        eval_metric="quantile",
        categorical_feature=list(CATEGORICAL_INDICES),
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model


def labeled_rows(
    features: FloatArray, values: FloatArray, weights: FloatArray, clusters: FloatArray
) -> TrainingData:
    """Keep rows with a target value and a positive survey weight.

    A missing or zero weight means the person was not in the subsample that
    measured this target, so the row cannot represent anyone.

    Args:
        features: Feature matrix for every row.
        values: Target values, NaN where missing.
        weights: Survey weights, NaN where missing.
        clusters: NHANES design cluster per row.

    Returns:
        The usable rows.
    """
    keep = ~np.isnan(values) & np.isfinite(weights) & (weights > 0)
    return TrainingData(features[keep], values[keep], weights[keep], clusters[keep])


def cross_validate(
    data: TrainingData,
    quantiles: tuple[float, ...] = QUANTILES,
    param_grid: list[dict[str, int]] | None = None,
) -> CVResult:
    """Pick the best configuration and per-quantile tree counts with grouped K-fold CV.

    Folds are grouped by NHANES design cluster, so people from one sampled area are
    never in both training and validation. Losses are survey-weighted, so the choice
    reflects the US population. The best configuration is then refit per fold with
    its chosen tree counts to give honest out-of-fold predictions for calibration.

    Args:
        data: Training rows.
        quantiles: Quantile levels to fit.
        param_grid: Configurations to compare. Defaults to model_config.PARAM_GRID.

    Returns:
        The best configuration, its tree counts and out-of-fold predictions.
    """
    grid = param_grid or model_config.PARAM_GRID
    split_rows = list(
        GroupKFold(n_splits=model_config.N_SPLITS).split(
            data.features, groups=data.clusters
        )
    )
    folds = [
        (select_rows(data, train_rows), select_rows(data, valid_rows))
        for train_rows, valid_rows in split_rows
    ]
    losses: defaultdict[tuple[int, float], list[float]] = defaultdict(list)
    best_iterations: defaultdict[tuple[int, float], list[int]] = defaultdict(list)

    for cfg_index, cfg in enumerate(grid):
        for tau in quantiles:
            for train, validation in folds:
                model = fit_with_early_stopping(make_lgbm(tau, cfg), train, validation)
                prediction = np.asarray(
                    model.predict(validation.features), dtype=np.float64
                )
                losses[cfg_index, tau].append(
                    pinball_loss(validation.values, prediction, tau, validation.weights)
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
    n_estimators_by_tau = {
        tau: max(MIN_N_ESTIMATORS, round(float(np.mean(best_iterations[best, tau]))))
        for tau in quantiles
    }
    oof = np.empty((len(data.values), len(quantiles)))
    fold_ids = np.empty(len(data.values), dtype=np.int_)
    for fold, ((train, validation), (_, valid_rows)) in enumerate(
        zip(folds, split_rows, strict=True)
    ):
        models = fit_final(train, grid[best], n_estimators_by_tau, quantiles)
        oof[valid_rows] = predict_quantiles(models, validation.features, quantiles)
        fold_ids[valid_rows] = fold
    return CVResult(
        config_name=config_name(grid[best]),
        cfg=grid[best],
        n_estimators_by_tau=n_estimators_by_tau,
        pinball_mean_by_tau={
            tau: float(np.mean(losses[best, tau])) for tau in quantiles
        },
        mean_pinball=mean_by_cfg[best],
        oof_predictions=oof,
        fold_ids=fold_ids,
    )


def fit_final(
    data: TrainingData,
    cfg: Mapping[str, int],
    n_estimators_by_tau: Mapping[float, int],
    quantiles: tuple[float, ...] = QUANTILES,
) -> QuantileModels:
    """Fit one survey-weighted quantile model per tau on all given rows.

    Args:
        data: Training rows.
        cfg: LightGBM parameters.
        n_estimators_by_tau: Tree count per quantile.
        quantiles: Quantile levels to fit.

    Returns:
        Fitted models keyed by tau.
    """
    models: QuantileModels = {}
    for tau in quantiles:
        model = make_lgbm(tau, {**cfg, "n_estimators": n_estimators_by_tau[tau]})
        model.fit(
            data.features,
            data.values,
            sample_weight=data.weights,
            categorical_feature=list(CATEGORICAL_INDICES),
        )
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
