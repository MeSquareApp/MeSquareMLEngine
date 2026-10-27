"""Shared config for quantile biomarker models (training and inference)."""

from __future__ import annotations

TARGETS: tuple[str, ...] = ("non_hdl", "hdl", "hba1c", "egfr")

FEATURES: tuple[str, ...] = (
    "sex",
    "age",
    "race_eth",
    "weight_kg",
    "waist_cm",
    "smoking",
    "told_high_bp",
    "bp_meds",
    "told_high_chol",
    "chol_meds",
    "diabetes_dx",
    "prediabetes_dx",
    "insulin",
    "diabetes_pills",
)

CATEGORICAL: tuple[str, ...] = ("sex", "race_eth", "smoking")

QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

QUANTILE_LABELS: dict[float, str] = {
    0.10: "p10",
    0.50: "p50",
    0.90: "p90",
}

TARGET_UNITS: dict[str, str] = {
    "non_hdl": "mg/dL",
    "hdl": "mg/dL",
    "hba1c": "%",
    "egfr": "mL/min/1.73m2",
}

# NHANES RIDRETH3 codes used as categorical levels for race_eth.
# Sex: RIAGENDR 1=male, 2=female. Smoking: derived labels from SMQ.
DEFAULT_CATEGORY_LEVELS: dict[str, list] = {
    "sex": [1.0, 2.0],
    "race_eth": [1.0, 2.0, 3.0, 4.0, 6.0, 7.0],
    "smoking": ["never", "former", "current"],
}

DIABETES_DX_MAP: dict[str, float] = {
    "no": 0.0,
    "yes": 1.0,
    "borderline": 0.5,
}

PARAM_GRID: list[dict[str, int]] = [
    {"num_leaves": 15, "min_child_samples": 20},
    {"num_leaves": 15, "min_child_samples": 50},
    {"num_leaves": 31, "min_child_samples": 20},
    {"num_leaves": 31, "min_child_samples": 50},
]

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.2
MIN_N_ESTIMATORS = 50
DEFAULT_N_ESTIMATORS = 2000
LEARNING_RATE = 0.03
EARLY_STOPPING_ROUNDS = 100
MAX_PREDICT_BATCH = 1000
