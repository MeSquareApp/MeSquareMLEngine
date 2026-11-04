"""Offline training for quantile biomarker models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

from config.settings import Settings, settings
from src.components.analytes import ANALYTES
from src.components.calibration import (
    calibration_groups,
    cross_conformal_covered,
    mondrian_widening,
    subgroup_coverage,
    widening_per_row,
)
from src.components.egfr import with_egfr
from src.components.feature_engineering import infer_category_levels, prepare_features
from src.components.model_artifact import QuantileModelBundle, build_bundle
from src.components.model_config import (
    MIN_TRAINING_ROWS,
    QUANTILES,
    RANDOM_STATE,
    TARGET_WEIGHTS,
    TARGETS,
    TEST_SIZE,
)
from src.components.model_evaluation import (
    EvalContext,
    FloatArray,
    MetricRow,
    evaluate_target,
    interval_columns,
    widen_interval,
)
from src.components.model_trainer import (
    QuantileModels,
    cross_validate,
    fit_final,
    labeled_rows,
    predict_quantiles,
    select_rows,
)
from src.repository.postgres_db_connection import NhanesRepository
from src.utils.exception import MeSquareError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# wt_lab (phlebotomy weight) is in the training table. The exam weight that SBP needs
# comes from the raw demographics table.
TRAINING_QUERY = """
SELECT t.*, d.wtmec2yr AS wt_mec
FROM ml.cardiovascular_training AS t
JOIN nhanes.demographics AS d USING (seqn)
"""


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
        interval_widening: Conformal widening of the p10-p90 interval per
            calibration group (sex x age band), plus the pooled value.
        subgroups: Coverage by sex and by age band with the largest gap, from
            cross-conformal evaluation over every training row ("cross_conformal")
            and from the untouched holdout ("holdout").
    """

    models: QuantileModels
    eval_rows: list[MetricRow]
    interval_row: MetricRow
    cv_summary: dict[str, object]
    n_rows: int
    interval_widening: dict[str, float]
    subgroups: dict[str, dict[str, dict[str, float]]]


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


def design_clusters(frame: pl.DataFrame) -> FloatArray:
    """NHANES design cluster (stratum x PSU) of every row.

    Args:
        frame: Training rows with sdmvstra and sdmvpsu.

    Returns:
        One cluster id per row.
    """
    return (
        frame.select(pl.col("sdmvstra") * 100 + pl.col("sdmvpsu"))
        .to_series()
        .cast(pl.Float64)
        .fill_null(-1.0)
        .to_numpy()
    )


def plausible_values(values: FloatArray, target: str) -> FloatArray:
    """Outlier rule: values outside the analyte's plausible range become missing.

    The range is the one lab ingestion uses, so training never learns from a value
    the API would reject.

    Args:
        values: Target values, NaN where missing.
        target: Analyte name.

    Returns:
        A copy with implausible values set to NaN.
    """
    low, high = ANALYTES[target].plausible
    return np.where((values >= low) & (values <= high), values, np.nan)


def train_target(
    features: FloatArray,
    values: FloatArray,
    weights: FloatArray,
    clusters: FloatArray,
    target: str,
) -> TargetTraining:
    """Tune, calibrate, evaluate on a grouped holdout, then refit on every row.

    Fitting, early stopping, CV losses, calibration and holdout metrics all use the
    NHANES survey weights, so the quantiles describe US adults. The holdout and the
    CV folds are split by design cluster. The p10-p90 interval is calibrated per
    sex x age band on out-of-fold predictions, then judged on the untouched holdout.

    Args:
        features: Feature matrix for every training row.
        values: Target values, NaN where missing or implausible.
        weights: Survey weights for this target, NaN where missing.
        clusters: NHANES design cluster per row.
        target: Biomarker name.

    Returns:
        Production models and metrics.

    Raises:
        TrainingError: If fewer than MIN_TRAINING_ROWS rows are usable.
    """
    data = labeled_rows(features, values, weights, clusters)
    if len(data.values) < MIN_TRAINING_ROWS:
        raise TrainingError(f"Not enough labeled rows for {target}: {len(data.values)}")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_rows, test_rows = next(splitter.split(data.features, groups=data.clusters))
    train, test = select_rows(data, train_rows), select_rows(data, test_rows)
    cv = cross_validate(train)
    logger.info(
        "best for %s: %s (avg pinball=%.4f)", target, cv.config_name, cv.mean_pinball
    )

    train_groups = calibration_groups(train.features)
    widening = mondrian_widening(
        train.values, cv.oof_predictions, train.weights, train_groups, QUANTILES
    )
    cross_conformal = subgroup_coverage(
        cross_conformal_covered(
            train.values, cv.oof_predictions, train.weights, train_groups, cv.fold_ids
        ),
        train.weights,
        train_groups,
    )
    test_groups = calibration_groups(test.features)
    holdout_models = fit_final(train, cv.cfg, cv.n_estimators_by_tau)
    test_pred = widen_interval(
        predict_quantiles(holdout_models, test.features),
        QUANTILES,
        widening_per_row(test_groups, widening),
    )
    eval_rows, interval_row = evaluate_target(
        (test.values, test.weights),
        test_pred,
        (train.values, train.weights),
        QUANTILES,
        EvalContext(target, cv.config_name, cv.n_estimators_by_tau),
    )
    lower, upper = interval_columns(QUANTILES)
    covered = (
        (test.values >= test_pred[:, lower]) & (test.values <= test_pred[:, upper])
    ).astype(np.float64)
    subgroups = {
        "cross_conformal": cross_conformal,
        "holdout": subgroup_coverage(covered, test.weights, test_groups),
    }
    logger.info(
        "%s holdout coverage=%.3f width=%.2f; cross-conformal gap sex=%.1f "
        "age=%.1f pts",
        target,
        interval_row["coverage"],
        interval_row["mean_width"],
        cross_conformal["sex"]["gap"],
        cross_conformal["age_band"]["gap"],
    )
    return TargetTraining(
        models=fit_final(data, cv.cfg, cv.n_estimators_by_tau),
        eval_rows=eval_rows,
        interval_row=interval_row,
        cv_summary={
            "config": cv.config_name,
            "cfg": cv.cfg,
            "mean_pinball": cv.mean_pinball,
            "n_estimators_by_tau": cv.n_estimators_by_tau,
            "pinball_mean_by_tau": cv.pinball_mean_by_tau,
        },
        n_rows=len(data.values),
        interval_widening=widening,
        subgroups=subgroups,
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
    clusters = design_clusters(raw)

    trained: dict[str, TargetTraining] = {}
    exclusions: dict[str, dict[str, int]] = {}
    for target in TARGETS:
        logger.info("=== Training target: %s ===", target)
        raw_values, weights = (
            raw[column].cast(pl.Float64).fill_null(np.nan).to_numpy()
            for column in (target, TARGET_WEIGHTS[target])
        )
        values = plausible_values(raw_values, target)
        exclusions[target] = {
            "rows": len(values),
            "missing_value": int(np.isnan(raw_values).sum()),
            "implausible_value": int(
                np.isnan(values).sum() - np.isnan(raw_values).sum()
            ),
        }
        trained[target] = train_target(features, values, weights, clusters, target)

    bundle = build_bundle(
        models={target: result.models for target, result in trained.items()},
        category_levels=category_levels,
        metrics={
            "eval": [row for result in trained.values() for row in result.eval_rows],
            "intervals": [result.interval_row for result in trained.values()],
            "cv": {target: result.cv_summary for target, result in trained.items()},
            "subgroups": {
                target: result.subgroups for target, result in trained.items()
            },
            "exclusions": exclusions,
        },
        training_row_counts={
            target: result.n_rows for target, result in trained.items()
        },
    )
    bundle = replace(
        bundle,
        interval_widening={
            target: result.interval_widening for target, result in trained.items()
        },
    )
    saved = bundle.save(Path(model_dir or app_settings.model_dir))
    logger.info("Saved quantile models to %s (version=%s)", saved, bundle.version)
    return bundle
