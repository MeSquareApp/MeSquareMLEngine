"""AHA PREVENT 10-year risk, computed by the official AHA R package.

All risk numbers come from `AHAprevent::prevent_base()` (AHA-DS-Analytics/PREVENT, v1.1.0),
called in-process through rpy2. This module only converts inputs and outputs. It never
computes risk itself, and there is no fallback when R or the package is missing.

Install the package: R CMD INSTALL AHAprevent_1.1.0.tar.gz (from the PREVENT repo).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.utils.exception import MeSquareError

PACKAGE = "AHAprevent"
SEX_MALE = 1
SEX_FEMALE = 2
AGE_RANGE = (30, 79)
OUTCOMES = ("total_cvd", "ascvd", "heart_failure")
# Columns of prevent_base() for each outcome, 10-year horizon.
_OUTPUT_COLUMNS = {
    "total_cvd": "prevent_base_10yr_CVD",
    "ascvd": "prevent_base_10yr_ASCVD",
    "heart_failure": "prevent_base_10yr_HF",
}
# Validated input ranges from aha_prevent_base.ado. Outside them, the Stata package returns missing.
VALIDATED_RANGES = {
    "total_chol": (130.0, 320.0),
    "hdl": (20.0, 100.0),
    "sbp": (90.0, 200.0),
    "bmi": (18.5, 39.99),
}

# The embedded R interpreter is not thread-safe, and FastAPI runs sync endpoints in a threadpool.
_R_LOCK = threading.Lock()


class PreventUnavailableError(MeSquareError):
    """Raised when R or the AHAprevent package cannot be loaded."""


@dataclass(frozen=True)
class PreventInputs:
    """Cholesterol in mg/dL, SBP in mmHg, BMI in kg/m2, eGFR in mL/min/1.73m2.

    Sex uses NHANES codes (1 = male, 2 = female). Flags are 0/1. Any field may be an array of
    samples.
    """

    age: float
    sex: int
    total_chol: np.ndarray | float
    hdl: np.ndarray | float
    sbp: np.ndarray | float
    bmi: np.ndarray | float
    egfr: np.ndarray | float
    diabetes: np.ndarray | float
    smoking: float
    bp_treated: float
    statin: float


@lru_cache(maxsize=1)
def _package():
    try:
        from rpy2.robjects import default_converter
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr
    except Exception as error:  # rpy2 raises various errors when R itself is missing
        raise PreventUnavailableError(f"rpy2/R is not available: {error}") from error
    try:
        with _R_LOCK, localconverter(default_converter):
            return importr(PACKAGE)
    except Exception as error:
        raise PreventUnavailableError(f"R package {PACKAGE} is not installed: {error}") from error


def init_prevent() -> bool:
    """Start R and load the package. Call once from the main thread at app startup."""
    return prevent_available()


def prevent_available() -> bool:
    try:
        _package()
    except PreventUnavailableError:
        return False
    return True


def prevent_risk(inputs: PreventInputs) -> dict[str, np.ndarray]:
    """10-year risks as fractions, one per sample. Raises ValueError outside ages 30-79."""
    if not AGE_RANGE[0] <= inputs.age <= AGE_RANGE[1]:
        raise ValueError(f"PREVENT is valid for ages {AGE_RANGE[0]}-{AGE_RANGE[1]}")
    if inputs.sex not in (SEX_MALE, SEX_FEMALE):
        raise ValueError(f"Unsupported sex code: {inputs.sex!r}")

    columns = {
        "tc": inputs.total_chol,
        "hdl": inputs.hdl,
        "sbp": inputs.sbp,
        "dm": inputs.diabetes,
        "smoking": inputs.smoking,
        "bmi": inputs.bmi,
        "egfr": inputs.egfr,
        "bptreat": inputs.bp_treated,
        "statin": inputs.statin,
    }
    n = max(np.size(value) for value in columns.values())
    vectors = {name: np.broadcast_to(np.asarray(value, dtype=float), (n,)) for name, value in columns.items()}
    # The official package codes sex as 0 = male, 1 = female.
    sex = np.full(n, 0.0 if inputs.sex == SEX_MALE else 1.0)
    age = np.full(n, float(inputs.age))

    from rpy2.robjects import FloatVector, default_converter
    from rpy2.robjects.conversion import localconverter

    package = _package()
    # Conversion rules live in a ContextVar that worker threads do not inherit, so set them here.
    with _R_LOCK, localconverter(default_converter):
        result = package.prevent_base(
            FloatVector(sex),
            FloatVector(age),
            *(FloatVector(vectors[name]) for name in (
                "tc", "hdl", "sbp", "dm", "smoking", "bmi", "egfr", "bptreat", "statin"
            )),
        )
        # prevent_base() builds its data frame from a Vectorize() matrix, so each column is a
        # list of length-1 numeric vectors. NULL or NA cells become NaN.
        frame = {
            name: np.array([float(cell[0]) if len(cell) else np.nan for cell in result.rx2(name)])
            for name in _OUTPUT_COLUMNS.values()
        }
    # The package reports percentages. The engine works in fractions.
    return {outcome: frame[column] / 100.0 for outcome, column in _OUTPUT_COLUMNS.items()}


def clip_to_validated(name: str, values):
    """Clip posterior samples into the validated range so no draw is dropped as missing."""
    low, high = VALIDATED_RANGES[name]
    return np.clip(values, low, high)


def summarize_risk(samples: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Median and 80% interval per outcome from Monte-Carlo risk samples."""
    return {
        outcome: {
            "median": float(np.median(values)),
            "q10": float(np.quantile(values, 0.10)),
            "q90": float(np.quantile(values, 0.90)),
        }
        for outcome, values in ((o, np.atleast_1d(samples[o])) for o in OUTCOMES)
    }
