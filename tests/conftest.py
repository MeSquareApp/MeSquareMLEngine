"""Shared fixtures for quantile prediction tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.components.feature_engineering import prepare_features
from src.components.model_artifact import QuantileModelBundle, build_bundle
from src.components.model_config import DEFAULT_CATEGORY_LEVELS, QUANTILES, TARGETS
from src.components.model_trainer import fit_final
from src.pipeline.prediction_pipeline import QuantilePredictor


def _synthetic_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sex = rng.choice([1.0, 2.0], size=n)
    age = rng.uniform(20, 79, size=n)
    race_eth = rng.choice([1.0, 2.0, 3.0, 4.0, 6.0, 7.0], size=n)
    weight = rng.normal(80, 15, size=n).clip(40, 160)
    waist = rng.normal(95, 12, size=n).clip(60, 150)
    smoking = rng.choice(["never", "former", "current"], size=n)
    diabetes_dx = rng.choice(["no", "yes", "borderline"], size=n, p=[0.7, 0.2, 0.1])

    # Simple signal so LightGBM can learn something.
    non_hdl = 100 + 0.4 * age + 0.3 * waist + rng.normal(0, 8, size=n)
    hdl = 55 - 0.1 * waist + 5 * (sex == 2.0) + rng.normal(0, 4, size=n)
    hba1c = 5.2 + 0.01 * age + 0.8 * (diabetes_dx == "yes") + rng.normal(0, 0.3, size=n)
    egfr = 120 - 0.6 * age + rng.normal(0, 5, size=n)

    return pd.DataFrame(
        {
            "sex": sex,
            "age": age,
            "race_eth": race_eth,
            "weight_kg": weight,
            "waist_cm": waist,
            "smoking": smoking,
            "told_high_bp": rng.integers(0, 2, size=n),
            "bp_meds": rng.integers(0, 2, size=n),
            "told_high_chol": rng.integers(0, 2, size=n),
            "chol_meds": rng.integers(0, 2, size=n),
            "diabetes_dx": diabetes_dx,
            "prediabetes_dx": rng.integers(0, 2, size=n),
            "insulin": rng.integers(0, 2, size=n),
            "diabetes_pills": rng.integers(0, 2, size=n),
            "non_hdl": non_hdl,
            "hdl": hdl,
            "hba1c": hba1c,
            "egfr": egfr,
        }
    )


@pytest.fixture
def category_levels() -> dict[str, list]:
    return {k: list(v) for k, v in DEFAULT_CATEGORY_LEVELS.items()}


@pytest.fixture
def tiny_bundle(tmp_path: Path, category_levels: dict[str, list]) -> QuantileModelBundle:
    frame = _synthetic_frame()
    features = prepare_features(frame, category_levels)
    models = {}
    row_counts = {}
    cfg = {"num_leaves": 7, "min_child_samples": 5}
    n_estimators = {tau: 20 for tau in QUANTILES}
    for target in TARGETS:
        y = frame[target].astype(float).to_numpy()
        models[target] = fit_final(features, y, cfg, n_estimators, QUANTILES)
        row_counts[target] = len(y)
    bundle = build_bundle(
        models=models,
        category_levels=category_levels,
        metrics={"eval": []},
        training_row_counts=row_counts,
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
