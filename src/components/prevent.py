"""AHA PREVENT 10-year risk, computed by the official AHA R package.

All risk numbers come from `AHAprevent::prevent_base()` (AHA-DS-Analytics/PREVENT,
v1.1.0), called in-process through rpy2. This module only converts inputs and outputs.
It never computes risk itself, and there is no fallback when R or the package is
missing.

Install the package: R CMD INSTALL AHAprevent_1.1.0.tar.gz (from the PREVENT repo).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import cache
from types import ModuleType
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from src.utils.exception import MeSquareError
from src.utils.logger import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]

PACKAGE = "AHAprevent"
SEX_MALE = 1
SEX_FEMALE = 2
AGE_RANGE = (30, 79)
OUTCOMES = ("total_cvd", "ascvd", "heart_failure")
HORIZONS = (10, 30)
# PREVENT's 30-year equations are validated for ages 30-59 only.
THIRTY_YEAR_MAX_AGE = 59
# prevent_base() argument order after sex and age.
ARGUMENT_ORDER = (
    "tc",
    "hdl",
    "sbp",
    "dm",
    "smoking",
    "bmi",
    "egfr",
    "bptreat",
    "statin",
)
# prevent_base() output column for each (outcome, horizon in years).
OUTPUT_COLUMNS = {
    (outcome, horizon): f"prevent_base_{horizon}yr_{suffix}"
    for outcome, suffix in (
        ("total_cvd", "CVD"),
        ("ascvd", "ASCVD"),
        ("heart_failure", "HF"),
    )
    for horizon in HORIZONS
}
PERCENT = 100.0
# Validated ranges from aha_prevent_base.ado. Outside them Stata returns missing.
VALIDATED_RANGES = {
    "total_chol": (130.0, 320.0),
    "hdl": (20.0, 100.0),
    "sbp": (90.0, 200.0),
    "bmi": (18.5, 39.99),
}

# Embedded R is not thread-safe, and FastAPI runs sync endpoints in a threadpool.
_R_LOCK = threading.Lock()


class PreventUnavailableError(MeSquareError):
    """Raised when R or the AHAprevent package cannot be loaded."""


class _RPackage(Protocol):
    def prevent_base(self, *args: object) -> object: ...


@dataclass(frozen=True)
class PreventInputs:
    """Inputs to PREVENT. Any numeric field may be an array of Monte-Carlo samples.

    Attributes:
        age: Age in years.
        sex: NHANES code, 1 = male, 2 = female.
        total_chol: Total cholesterol in mg/dL.
        hdl: HDL cholesterol in mg/dL.
        sbp: Systolic blood pressure in mmHg.
        bmi: Body-mass index in kg/m2.
        egfr: eGFR in mL/min/1.73m2.
        diabetes: 1 if diabetic, else 0.
        smoking: 1 if current smoker, else 0.
        bp_treated: 1 if on antihypertensives, else 0.
        statin: 1 if on a statin, else 0.
    """

    age: float
    sex: int
    total_chol: npt.ArrayLike
    hdl: npt.ArrayLike
    sbp: npt.ArrayLike
    bmi: npt.ArrayLike
    egfr: npt.ArrayLike
    diabetes: npt.ArrayLike
    smoking: float
    bp_treated: float
    statin: float


def _rpy2() -> ModuleType:
    """Import rpy2.robjects, which starts the embedded R interpreter.

    Raises:
        PreventUnavailableError: If rpy2 or R cannot be loaded.
    """
    try:
        import rpy2.robjects as robjects
    except Exception as error:
        # cffi raises ffi.error when libR is too old for this rpy2 build.
        raise PreventUnavailableError(f"rpy2/R is not available: {error}") from error
    # rpy2 ships no type information, so mypy sees the module as Any.
    return cast(ModuleType, robjects)


@cache
def _package() -> _RPackage:
    """Load AHAprevent once. Later calls reuse it.

    Raises:
        PreventUnavailableError: If R, rpy2, or the package is missing.
    """
    robjects = _rpy2()
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import PackageNotInstalledError, importr

    try:
        with _R_LOCK, localconverter(robjects.default_converter):
            package: _RPackage = importr(PACKAGE)
    except PackageNotInstalledError as error:
        raise PreventUnavailableError(
            f"R package {PACKAGE} is not installed"
        ) from error
    return package


@cache
def prevent_available() -> bool:
    """Whether the official package can be used. Logs the reason once when not.

    Call it once from the main thread at app startup, because embedded R must start
    there.

    Returns:
        True when R, rpy2, and AHAprevent load.
    """
    try:
        _package()
    except PreventUnavailableError as error:
        logger.error("AHA PREVENT unavailable: %s", error)
        return False
    return True


@cache
def package_version() -> str:
    """Installed AHAprevent version, for model_versions.

    Returns:
        For example "AHAprevent 1.1.0", or "unavailable" when R cannot load it.
    """
    if not prevent_available():
        return "unavailable"
    robjects = _rpy2()
    from rpy2.robjects.conversion import localconverter

    with _R_LOCK, localconverter(robjects.default_converter):
        version = robjects.r(f'as.character(packageVersion("{PACKAGE}"))')[0]
    return f"{PACKAGE} {version}"


def _call_prevent_base(
    sex: FloatArray, age: FloatArray, columns: dict[str, FloatArray]
) -> dict[str, FloatArray]:
    """Run AHAprevent::prevent_base() on equal-length vectors.

    Args:
        sex: Official sex codes, 0 = male, 1 = female.
        age: Age in years.
        columns: Remaining arguments keyed as in ARGUMENT_ORDER.

    Returns:
        10- and 30-year risks in percent, keyed by output column name.

    Raises:
        PreventUnavailableError: If R or the package is missing.
    """
    robjects = _rpy2()
    from rpy2.robjects.conversion import localconverter

    package = _package()
    vectors = [robjects.FloatVector(sex), robjects.FloatVector(age)]
    vectors += [robjects.FloatVector(columns[name]) for name in ARGUMENT_ORDER]
    # Conversion rules live in a ContextVar that worker threads do not inherit.
    with _R_LOCK, localconverter(robjects.default_converter):
        result = package.prevent_base(*vectors)
        # prevent_base() builds its frame from a Vectorize() matrix, so each column
        # is a list of length-1 vectors. Empty cells become NaN.
        return {
            column: np.array(
                [
                    float(cell[0]) if len(cell) else np.nan
                    for cell in result.rx2(column)  # type: ignore[attr-defined]
                ]
            )
            for column in OUTPUT_COLUMNS.values()
        }


def prevent_risk(inputs: PreventInputs) -> dict[tuple[str, int], FloatArray]:
    """PREVENT risks from the official package, one value per input row.

    Example:
        >>> risk = prevent_risk(PreventInputs(age=50, sex=2, total_chol=200, hdl=45,
        ...     sbp=160, bmi=35, egfr=90, diabetes=1, smoking=0, bp_treated=1,
        ...     statin=0))
        >>> round(float(risk["total_cvd", 10][0]), 3)
        0.147

    Args:
        inputs: Patient values. Array fields must share one length or be scalars.

    Returns:
        Risks as fractions keyed by (outcome, horizon in years). 30-year values
        are NaN above THIRTY_YEAR_MAX_AGE.

    Raises:
        ValueError: If age is outside 30-79 or sex is not 1 or 2.
        PreventUnavailableError: If R or the package is missing.
    """
    if not AGE_RANGE[0] <= inputs.age <= AGE_RANGE[1]:
        raise ValueError(f"PREVENT is valid for ages {AGE_RANGE[0]}-{AGE_RANGE[1]}")
    if inputs.sex not in (SEX_MALE, SEX_FEMALE):
        raise ValueError(f"Unsupported sex code: {inputs.sex!r}")

    raw: dict[str, npt.ArrayLike] = {
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
    n_samples = max(np.size(value) for value in raw.values())
    columns = {
        name: np.broadcast_to(np.asarray(value, dtype=np.float64), (n_samples,))
        for name, value in raw.items()
    }
    sex = np.full(n_samples, 0.0 if inputs.sex == SEX_MALE else 1.0)
    age = np.full(n_samples, float(inputs.age))
    percent = _call_prevent_base(sex, age, columns)
    risks = {key: percent[column] / PERCENT for key, column in OUTPUT_COLUMNS.items()}
    if inputs.age > THIRTY_YEAR_MAX_AGE:
        for outcome in OUTCOMES:
            risks[outcome, 30] = np.full(n_samples, np.nan)
    return risks


def clip_to_validated(name: str, values: npt.ArrayLike) -> FloatArray:
    """Clip samples into PREVENT's validated range so no draw is dropped as missing.

    Args:
        name: Key of VALIDATED_RANGES.
        values: Samples in the variable's unit.

    Returns:
        Clipped samples.

    Raises:
        KeyError: If name has no validated range.
    """
    low, high = VALIDATED_RANGES[name]
    return np.clip(np.asarray(values, dtype=np.float64), low, high)
