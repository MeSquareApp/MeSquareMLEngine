"""Judge the lab stage by simulation and write the activation registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

from config.settings import Settings, settings
from src.components.analytes import MODEL_ANALYTES
from src.components.calibration import calibration_groups, widening_per_row
from src.components.feature_engineering import prepare_features
from src.components.model_artifact import QuantileModelBundle
from src.components.model_config import RANDOM_STATE, TARGET_WEIGHTS, TEST_SIZE
from src.components.model_evaluation import FloatArray, widen_interval
from src.components.model_trainer import TrainingData, fit_final, predict_quantiles
from src.components.personalization.activation import (
    LAB_ONLY,
    REGISTRY_PATH,
    WEARABLE_ACTIVATION_ORDER,
    ActivationRegistry,
    StageDecision,
    save_registry,
)
from src.components.personalization.simulation import (
    SimulationInput,
    simulate_lab_stage,
)
from src.pipeline.prediction_pipeline import QuantilePredictor
from src.pipeline.training_pipeline import (
    design_clusters,
    load_training_frame,
    plausible_values,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

WEARABLE_PENDING = (
    "No real consented wearable-to-lab pairs yet; adjustments run in shadow mode "
    "until gate_residual_model() passes on them."
)


def grouped_split(
    frame: pl.DataFrame, analyte: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Training and held-out rows of one analyte, split exactly as training split it.

    Args:
        frame: Training frame from load_training_frame().
        analyte: Target analyte.

    Returns:
        (train, test) rows with a plausible value and a positive survey weight.
    """
    values = plausible_values(
        frame[analyte].cast(pl.Float64).fill_null(np.nan).to_numpy(), analyte
    )
    weights = (
        frame[TARGET_WEIGHTS[analyte]].cast(pl.Float64).fill_null(np.nan).to_numpy()
    )
    usable = ~np.isnan(values) & np.isfinite(weights) & (weights > 0)
    rows = frame.filter(pl.Series(usable))
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train, test = next(
        splitter.split(np.zeros(rows.height), groups=design_clusters(rows))
    )
    train_rows: pl.DataFrame = rows[train]
    test_rows: pl.DataFrame = rows[test]
    return train_rows, test_rows


def held_out_rows(frame: pl.DataFrame, analyte: str) -> pl.DataFrame:
    """The grouped holdout of one analyte.

    Args:
        frame: Training frame from load_training_frame().
        analyte: Target analyte.

    Returns:
        Held-out rows with a plausible value and a positive survey weight.
    """
    return grouped_split(frame, analyte)[1]


def holdout_predictions(
    bundle: QuantileModelBundle, train: pl.DataFrame, test: pl.DataFrame, analyte: str
) -> FloatArray:
    """Calibrated holdout quantiles from the bundle's tuned settings, fit on train only.

    The production bundle is refit on every row, so scoring it on the holdout would
    flatter it. This refits the tuned configuration on the training split and
    applies the bundle's group widening, as training did when it scored the holdout.

    Args:
        bundle: Production bundle (for configuration, levels and widening).
        train: Training rows from grouped_split().
        test: Held-out rows from grouped_split().
        analyte: Target analyte.

    Returns:
        (p10, p50, p90) per held-out row.
    """
    cv = bundle.metrics["cv"][analyte]  # type: ignore[index]
    data = TrainingData(
        prepare_features(train, bundle.category_levels),
        train[analyte].cast(pl.Float64).to_numpy(),
        train[TARGET_WEIGHTS[analyte]].cast(pl.Float64).to_numpy(),
        design_clusters(train),
    )
    trees = {float(tau): int(n) for tau, n in cv["n_estimators_by_tau"].items()}
    models = fit_final(data, cv["cfg"], trees, bundle.quantiles)
    features = prepare_features(test, bundle.category_levels)
    return widen_interval(
        predict_quantiles(models, features, bundle.quantiles),
        bundle.quantiles,
        widening_per_row(
            calibration_groups(features), bundle.interval_widening.get(analyte, {})
        ),
    )


def evaluate_lab_stage(
    app_settings: Settings | None = None, registry_path: Path = REGISTRY_PATH
) -> ActivationRegistry:
    """Simulate repeat labs for every analyte and write the activation registry.

    Args:
        app_settings: Settings with DATABASE_URL and model_dir.
        registry_path: Where to write the registry.

    Returns:
        The registry that was written.
    """
    app_settings = app_settings or settings
    bundle = QuantileModelBundle.load_latest(app_settings.model_dir)
    predictor = QuantilePredictor(bundle)
    frame = load_training_frame(app_settings)
    lab: dict[str, StageDecision] = {}
    for analyte in MODEL_ANALYTES:
        rows = held_out_rows(frame, analyte)
        population = predictor.quantile_matrices(rows.to_dicts())[analyte]
        metrics = simulate_lab_stage(
            SimulationInput(
                analyte=analyte,
                truth=rows[analyte].cast(pl.Float64).to_numpy(),
                population=population,
                weights=rows[TARGET_WEIGHTS[analyte]].cast(pl.Float64).to_numpy(),
            )
        )
        logger.info("%s lab stage: %s", analyte, metrics)
        lab[analyte] = StageDecision(
            active=bool(metrics["beats_population"]),
            evidence={"method": "simulation", "prior": bundle.version, **metrics},
        )
    wearable = {
        analyte: StageDecision(
            active=False,
            evidence={
                "reason": "eGFR stays lab-only"
                if analyte in LAB_ONLY
                else WEARABLE_PENDING,
                "eligible": analyte in WEARABLE_ACTIVATION_ORDER,
            },
        )
        for analyte in MODEL_ANALYTES
    }
    registry = ActivationRegistry(
        version=f"activation-{bundle.version}", lab=lab, wearable=wearable
    )
    save_registry(registry, registry_path)
    logger.info("Wrote activation registry %s", registry_path)
    return registry
