"""Offline training for quantile biomarker models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from config.settings import Settings, settings
from src.components.egfr import with_egfr
from src.components.feature_engineering import infer_category_levels, prepare_features
from src.components.model_artifact import QuantileModelBundle, build_bundle
from src.components.model_config import (
    MIN_TRAINING_ROWS,
    QUANTILES,
    RANDOM_STATE,
    TARGETS,
    TEST_SIZE,
)
from src.components.model_evaluation import (
    EvalContext,
    FloatArray,
    MetricRow,
    evaluate_target,
)
from src.components.model_trainer import (
    QuantileModels,
    cross_validate,
    fit_final,
    labeled_rows,
    predict_quantiles,
)
from src.repository.postgres_db_connection import NhanesRepository
from src.utils.exception import MeSquareError
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRAINING_QUERY = "SELECT * FROM ml.cardiovascular_training"


class TrainingError(MeSquareError):
    """Raised when model training cannot complete."""


@dataclass(frozen=True)
class TargetTraining:
    """Production models and metrics for one target.

    Attributes:
        models: Models refit on every labeled row, keyed by tau.
        eval_rows: Holdout metrics per quantile.
        interval_row: Holdout p10-p90 interval metrics.
        cv_summary: Selected configuration and CV losses.
        n_rows: Labeled rows.
    """

    models: QuantileModels
    eval_rows: list[MetricRow]
    interval_row: MetricRow
    cv_summary: dict[str, object]
    n_rows: int


def load_training_frame(app_settings: Settings) -> pl.DataFrame:
    """Read ml.cardiovascular_training and add eGFR.

    Args:
        app_settings: Needs DATABASE_URL.

    Returns:
        Training rows with egfr and without scr.

    Raises:
        TrainingError: If DATABASE_URL is missing or the table is empty.
    """
    if not app_settings.database_url:
        raise TrainingError("DATABASE_URL is required for training")
    repository = NhanesRepository.from_settings(app_settings)
    try:
        with repository.engine.connect() as connection:
            frame = pl.read_database(TRAINING_QUERY, connection=connection)
    finally:
        repository.close()
    if frame.is_empty():
        raise TrainingError("ml.cardiovascular_training is empty")
    return with_egfr(frame).drop("scr", strict=False)


def train_target(
    features: FloatArray, values: FloatArray, target: str
) -> TargetTraining:
    """Tune, evaluate on a holdout split, then refit on every labeled row.

    Args:
        features: Feature matrix for every training row.
        values: Target values, NaN where missing.
        target: Biomarker name.

    Returns:
        Production models and metrics.

    Raises:
        TrainingError: If fewer than MIN_TRAINING_ROWS rows are labeled.
    """
    X, y = labeled_rows(features, values)
    if len(y) < MIN_TRAINING_ROWS:
        raise TrainingError(f"Not enough labeled rows for {target}: {len(y)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    cv = cross_validate(X_train, y_train)
    logger.info(
        "best for %s: %s (avg pinball=%.4f)", target, cv.config_name, cv.mean_pinball
    )

    holdout_models = fit_final(X_train, y_train, cv.cfg, cv.n_estimators_by_tau)
    eval_rows, interval_row = evaluate_target(
        y_test,
        predict_quantiles(holdout_models, X_test),
        y_train,
        QUANTILES,
        EvalContext(target, cv.config_name, cv.n_estimators_by_tau),
    )
    logger.info(
        "%s holdout: interval80 coverage=%.3f width=%.2f",
        target,
        interval_row["coverage"],
        interval_row["mean_width"],
    )
    return TargetTraining(
        models=fit_final(X, y, cv.cfg, cv.n_estimators_by_tau),
        eval_rows=eval_rows,
        interval_row=interval_row,
        cv_summary={
            "config": cv.config_name,
            "cfg": cv.cfg,
            "mean_pinball": cv.mean_pinball,
            "n_estimators_by_tau": cv.n_estimators_by_tau,
            "pinball_mean_by_tau": cv.pinball_mean_by_tau,
        },
        n_rows=len(y),
    )


def train_quantile_models(
    app_settings: Settings | None = None, model_dir: Path | None = None
) -> QuantileModelBundle:
    """Train every target, evaluate on holdout, refit, and save a new bundle.

    Args:
        app_settings: Settings with DATABASE_URL. Defaults to the app settings.
        model_dir: Where to save. Defaults to app_settings.model_dir.

    Returns:
        The saved bundle.

    Raises:
        TrainingError: If data is missing or a target has too few labeled rows.
    """
    app_settings = app_settings or settings
    raw = load_training_frame(app_settings)
    category_levels = infer_category_levels(raw)
    features = prepare_features(raw, category_levels)

    trained: dict[str, TargetTraining] = {}
    for target in TARGETS:
        logger.info("=== Training target: %s ===", target)
        values = raw[target].cast(pl.Float64).fill_null(np.nan).to_numpy()
        trained[target] = train_target(features, values, target)

    bundle = build_bundle(
        models={target: result.models for target, result in trained.items()},
        category_levels=category_levels,
        metrics={
            "eval": [row for result in trained.values() for row in result.eval_rows],
            "intervals": [result.interval_row for result in trained.values()],
            "cv": {target: result.cv_summary for target, result in trained.items()},
        },
        training_row_counts={
            target: result.n_rows for target, result in trained.items()
        },
    )
    saved = bundle.save(Path(model_dir or app_settings.model_dir))
    logger.info("Saved quantile models to %s (version=%s)", saved, bundle.version)
    return bundle
