"""Shared fixtures for quantile prediction tests."""

from __future__ import annotations

from datetime import date, timedelta
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
    sbp = 105 + 0.4 * age + 0.1 * weight + rng.normal(0, 8, size=n)

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
            "sbp": sbp,
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


# Population medians and SDs used by the stub prior in digital twin tests.
STUB_PRIOR = {
    "non_hdl": (150.0, 40.0),
    "hdl": (50.0, 14.0),
    "hba1c": (5.5, 0.6),
    "egfr": (90.0, 18.0),
    "sbp": (125.0, 16.0),
}
Z_SCORES = {0.10: -1.2816, 0.50: 0.0, 0.90: 1.2816}


class StubPrior:
    """Fixed normal quantiles, so twin assertions don't depend on a trained model."""

    version = "stub"

    def predict_quantiles(self, profile: pd.DataFrame) -> dict[str, dict[float, float]]:
        return {
            target: {tau: median + z * sd for tau, z in Z_SCORES.items()}
            for target, (median, sd) in STUB_PRIOR.items()
        }


@pytest.fixture
def stub_prior() -> StubPrior:
    return StubPrior()


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


def make_wearables(
    start: date, n_days: int, change_day: int | None = 45, seed: int = 0
) -> pd.DataFrame:
    """Daily wearables. From change_day: +4000 steps and weight falling 0.04 kg/day."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_days):
        active = change_day is not None and i >= change_day
        rows.append(
            {
                "date": start + timedelta(days=i),
                "steps": 5000 + (4000 if active else 0) + rng.normal(0, 500),
                "resting_hr": 70 + rng.normal(0, 1),
                "sleep_h": 7 + rng.normal(0, 0.3),
                "weight_kg": 92 - (0.04 * (i - change_day) if active else 0) + rng.normal(0, 0.2),
            }
        )
    return pd.DataFrame(rows)
