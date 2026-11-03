"""Offline training for quantile biomarker models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import Settings, settings
from src.components.egfr import add_egfr
from src.components.feature_engineering import infer_category_levels, prepare_features
from src.components.model_artifact import QuantileModelBundle, build_bundle
from src.components.model_config import QUANTILES, RANDOM_STATE, TARGETS, TEST_SIZE
from src.components.model_evaluation import evaluate_target
from src.components.model_trainer import (
    cross_validate,
    fit_final,
    prepare_xy,
    predict_quantiles,
)
from src.repository.postgres_db_connection import NhanesRepository
from src.utils.exception import MeSquareError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingError(MeSquareError):
    """Raised when model training cannot complete."""


def load_training_frame(app_settings: Settings) -> pd.DataFrame:
    if not app_settings.database_url:
        raise TrainingError("DATABASE_URL is required for training")
    repository = NhanesRepository.from_settings(app_settings)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM ml.cardiovascular_training",
            con=repository.engine,
        )
    finally:
        repository.close()
    if df.empty:
        raise TrainingError("ml.cardiovascular_training is empty")
    return add_egfr(df).drop(columns="scr", errors="ignore")


def train_quantile_models(
    app_settings: Settings | None = None,
    model_dir: Path | None = None,
) -> QuantileModelBundle:
    """Train per-target quantile models, evaluate on holdout, refit, and save."""
    app_settings = app_settings or settings
    out_root = Path(model_dir or app_settings.model_dir)

    raw = load_training_frame(app_settings)
    category_levels = infer_category_levels(raw)
    features = prepare_features(raw, category_levels)

    models: dict = {}
    metrics: dict = {"eval": [], "intervals": [], "cv": {}}
    training_row_counts: dict[str, int] = {}

    for target in TARGETS:
        logger.info("=== Training target: %s ===", target)
        X, y = prepare_xy(raw, target, features)
        training_row_counts[target] = int(len(y))
        if len(y) < 50:
            raise TrainingError(f"Not enough labeled rows for {target}: {len(y)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        cv = cross_validate(X_train, y_train)
        logger.info(
            "best for %s: %s (avg pinball=%.4f)",
            target,
            cv.config_name,
            cv.mean_pinball,
        )
        metrics["cv"][target] = {
            "config": cv.config_name,
            "cfg": cv.cfg,
            "mean_pinball": cv.mean_pinball,
            "n_estimators_by_tau": {str(k): v for k, v in cv.n_estimators_by_tau.items()},
            "pinball_mean_by_tau": {str(k): v for k, v in cv.pinball_mean_by_tau.items()},
        }

        # Holdout evaluation with models fit on the train split only.
        holdout_models = fit_final(
            X_train, y_train, cv.cfg, cv.n_estimators_by_tau, QUANTILES
        )
        pred_mat = predict_quantiles(holdout_models, X_test, QUANTILES)
        eval_rows, interval_row = evaluate_target(
            y_test=y_test,
            pred_matrix=pred_mat,
            y_train=y_train,
            quantiles=QUANTILES,
            target=target,
            config_name=cv.config_name,
            n_estimators_by_tau=cv.n_estimators_by_tau,
        )
        metrics["eval"].extend(eval_rows)
        metrics["intervals"].append(interval_row)
        logger.info(
            "%s holdout: interval80 coverage=%.3f width=%.2f",
            target,
            interval_row["coverage"],
            interval_row["mean_width"],
        )

        # Production models: refit on all labeled rows with tuned n_estimators.
        models[target] = fit_final(X, y, cv.cfg, cv.n_estimators_by_tau, QUANTILES)

    bundle = build_bundle(
        models=models,
        category_levels=category_levels,
        metrics=metrics,
        training_row_counts=training_row_counts,
    )
    saved = bundle.save(out_root)
    logger.info("Saved quantile models to %s (version=%s)", saved, bundle.version)
    return bundle
