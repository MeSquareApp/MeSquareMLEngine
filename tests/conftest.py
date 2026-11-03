"""Shared fixtures for quantile prediction tests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.components.feature_engineering import prepare_features
from src.components.model_artifact import QuantileModelBundle, build_bundle
from src.components.model_config import (
    DEFAULT_CATEGORY_LEVELS,
    QUANTILES,
    TARGETS,
    CategoryLevels,
)
from src.components.model_trainer import TrainingData, fit_final
from src.pipeline.prediction_pipeline import QuantilePredictor

TINY_CONFIG = {"num_leaves": 7, "min_child_samples": 5}
TINY_TREES = 20


def synthetic_frame(n_rows: int = 120, seed: int = 0) -> pl.DataFrame:
    """Training-shaped frame with a simple signal so LightGBM learns something."""
    rng = np.random.default_rng(seed)
    sex = rng.choice([1.0, 2.0], size=n_rows)
    age = rng.uniform(20, 79, size=n_rows)
    weight = rng.normal(80, 15, size=n_rows).clip(40, 160)
    waist = rng.normal(95, 12, size=n_rows).clip(60, 150)
    diabetes_dx = rng.choice(
        ["no", "yes", "borderline"], size=n_rows, p=[0.7, 0.2, 0.1]
    )
    flags = {
        name: rng.integers(0, 2, size=n_rows)
        for name in (
            "told_high_bp",
            "bp_meds",
            "told_high_chol",
            "chol_meds",
            "prediabetes_dx",
            "insulin",
            "diabetes_pills",
        )
    }
    return pl.DataFrame(
        {
            "sex": sex,
            "age": age,
            "race_eth": rng.choice([1.0, 2.0, 3.0, 4.0, 6.0, 7.0], size=n_rows),
            "weight_kg": weight,
            "waist_cm": waist,
            "smoking": rng.choice(["never", "former", "current"], size=n_rows),
            "diabetes_dx": diabetes_dx,
            **flags,
            "non_hdl": 100 + 0.4 * age + 0.3 * waist + rng.normal(0, 8, size=n_rows),
            "hdl": 55 - 0.1 * waist + 5 * (sex == 2.0) + rng.normal(0, 4, size=n_rows),
            "hba1c": 5.2
            + 0.01 * age
            + 0.8 * (diabetes_dx == "yes")
            + rng.normal(0, 0.3, size=n_rows),
            "egfr": 120 - 0.6 * age + rng.normal(0, 5, size=n_rows),
            "sbp": 105 + 0.4 * age + 0.1 * weight + rng.normal(0, 8, size=n_rows),
            "wt_lab": rng.uniform(1_000, 50_000, size=n_rows),
            "wt_mec": rng.uniform(1_000, 50_000, size=n_rows),
        }
    )


@pytest.fixture
def category_levels() -> CategoryLevels:
    return {name: list(levels) for name, levels in DEFAULT_CATEGORY_LEVELS.items()}


@pytest.fixture
def tiny_bundle(tmp_path: Path, category_levels: CategoryLevels) -> QuantileModelBundle:
    """Small bundle trained on synthetic data, saved to and loaded from tmp_path."""
    frame = synthetic_frame()
    features = prepare_features(frame, category_levels)
    n_estimators = dict.fromkeys(QUANTILES, TINY_TREES)
    models = {
        target: fit_final(
            TrainingData(
                features, frame[target].to_numpy(), frame["wt_lab"].to_numpy()
            ),
            TINY_CONFIG,
            n_estimators,
            QUANTILES,
        )
        for target in TARGETS
    }
    bundle = build_bundle(
        models=models,
        category_levels=category_levels,
        metrics={"eval": []},
        training_row_counts=dict.fromkeys(TARGETS, frame.height),
        version="testbundle",
    )
    bundle.save(tmp_path)
    return QuantileModelBundle.load(tmp_path / "testbundle")


@pytest.fixture
def predictor(tiny_bundle: QuantileModelBundle) -> QuantilePredictor:
    return QuantilePredictor(tiny_bundle)


@pytest.fixture
def sample_patient() -> dict:
    """API request shape: sex/race_eth as NHANES labels."""
    return {
        "sex": "Male",
        "age": 52.0,
        "race_eth": "Non-Hispanic White",
        "weight_kg": 88.0,
        "waist_cm": 101.0,
        "smoking": "former",
        "told_high_bp": 1,
        "bp_meds": 1,
        "told_high_chol": 0,
        "chol_meds": 0,
        "diabetes_dx": "no",
        "prediabetes_dx": 0,
        "insulin": 0,
        "diabetes_pills": 0,
    }


@pytest.fixture
def sample_patient_codes(sample_patient: dict) -> dict:
    """Feature dict after API enum serialization (numeric NHANES codes)."""
    return {**sample_patient, "sex": 1.0, "race_eth": 3.0}


# Population medians and SDs used by the stub prior in digital twin tests.
STUB_PRIOR = {
    "non_hdl": (150.0, 40.0),
    "hdl": (50.0, 14.0),
    "hba1c": (5.5, 0.6),
    "egfr": (90.0, 18.0),
    "sbp": (125.0, 16.0),
}
Z_SCORES = {0.10: -1.2816, 0.50: 0.0, 0.90: 1.2816}
STEP_INCREASE = 4000.0
WEIGHT_LOSS_PER_DAY = 0.04


class StubPrior:
    """Fixed normal quantiles, so twin assertions do not depend on a trained model."""

    version = "stub"

    def predict_quantiles(
        self, record: Mapping[str, str | float | int | None]
    ) -> dict[str, dict[float, float]]:
        """Return the same quantiles for every record.

        Args:
            record: Ignored patient record.

        Returns:
            Quantiles keyed by target, then tau.
        """
        return {
            target: {tau: median + z * sd for tau, z in Z_SCORES.items()}
            for target, (median, sd) in STUB_PRIOR.items()
        }


def fake_prevent_base(
    sex: np.ndarray, age: np.ndarray, columns: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Stand-in for the R call: percent risk that rises with SBP and cholesterol."""
    base = columns["sbp"] / 20.0 + columns["tc"] / 100.0 + sex
    return {
        "prevent_base_10yr_CVD": base,
        "prevent_base_10yr_ASCVD": base / 2.0,
        "prevent_base_10yr_HF": base / 4.0,
    }


@pytest.fixture
def stub_prior() -> StubPrior:
    return StubPrior()


@pytest.fixture
def mock_prevent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the official R package with fake_prevent_base."""
    monkeypatch.setattr("src.components.prevent._call_prevent_base", fake_prevent_base)
    monkeypatch.setattr("src.pipeline.twin_pipeline.prevent_available", lambda: True)


@pytest.fixture
def twin_profile() -> dict:
    """API request shape for /v1/twin/predict."""
    return {
        "sex": "Male",
        "age": 52,
        "race_eth": "Non-Hispanic White",
        "height_cm": 178.0,
        "weight_kg": 92.0,
        "smoking": "never",
        "bp_meds": 0,
        "chol_meds": 0,
        "diabetes_dx": "no",
    }


@pytest.fixture
def twin_profile_codes(twin_profile: dict) -> dict:
    """twin_profile after API enum serialization (numeric NHANES codes)."""
    return {**twin_profile, "sex": 1, "race_eth": 3}


def make_wearables(
    start: date, n_days: int, change_day: int | None = 45, seed: int = 0
) -> list[dict[str, date | float | None]]:
    """Daily wearables. From change_day: more steps and slowly falling weight.

    Args:
        start: First day.
        n_days: Number of days.
        change_day: Day the behaviour change starts. None for no change.
        seed: Random seed for day-to-day noise.

    Returns:
        One record per day.
    """
    rng = np.random.default_rng(seed)
    records: list[dict[str, date | float | None]] = []
    for day_index in range(n_days):
        active = change_day is not None and day_index >= change_day
        days_active = day_index - change_day if active and change_day is not None else 0
        records.append(
            {
                "date": start + timedelta(days=day_index),
                "steps": 5000 + (STEP_INCREASE if active else 0) + rng.normal(0, 500),
                "resting_hr": 70 + rng.normal(0, 1),
                "sleep_h": 7 + rng.normal(0, 0.3),
                "weight_kg": 92
                - WEIGHT_LOSS_PER_DAY * days_active
                + rng.normal(0, 0.2),
            }
        )
    return records
