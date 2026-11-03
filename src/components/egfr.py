"""Race-free CKD-EPI 2021 estimated glomerular filtration rate.

PREVENT was derived with the 2021 equation, so both the prior's eGFR target and the
eGFR passed to PREVENT use it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

FloatArray = npt.NDArray[np.float64]

SEX_MALE = 1
SEX_FEMALE = 2
EGFR_INPUTS = ("scr", "age", "sex")


def egfr_ckd_epi_2021_array(
    scr: npt.ArrayLike, age: npt.ArrayLike, sex: npt.ArrayLike
) -> FloatArray:
    """Vectorized race-free CKD-EPI 2021 eGFR in mL/min/1.73m2.

    Equation: Inker et al., N Engl J Med 2021 (CKD-EPI creatinine, without race).

    Example:
        >>> round(float(egfr_ckd_epi_2021_array(1.0, 50, 1)), 1)
        91.7

    Args:
        scr: Serum creatinine in mg/dL.
        age: Age in years.
        sex: NHANES RIAGENDR codes, 1 = male, 2 = female.

    Returns:
        eGFR per row. NaN where any input is missing or sex is not 1 or 2.
    """
    sex_values = np.asarray(sex, dtype=np.float64)
    female = sex_values == SEX_FEMALE
    ratio = np.asarray(scr, dtype=np.float64) / np.where(female, 0.7, 0.9)
    egfr = (
        142.0
        * np.minimum(ratio, 1.0) ** np.where(female, -0.241, -0.302)
        * np.maximum(ratio, 1.0) ** -1.200
        * 0.9938 ** np.asarray(age, dtype=np.float64)
        * np.where(female, 1.012, 1.0)
    )
    return np.where(female | (sex_values == SEX_MALE), egfr, np.nan)


def egfr_ckd_epi_2021(scr: float, age: float, sex: int) -> float:
    """Race-free CKD-EPI 2021 eGFR in mL/min/1.73m2 for one person.

    Args:
        scr: Serum creatinine in mg/dL.
        age: Age in years.
        sex: NHANES RIAGENDR code, 1 = male, 2 = female.

    Returns:
        eGFR.

    Raises:
        ValueError: If sex is not 1 or 2.
    """
    if sex not in (SEX_MALE, SEX_FEMALE):
        raise ValueError(f"Unsupported sex code: {sex!r}")
    return float(egfr_ckd_epi_2021_array(scr, age, sex))


def with_egfr(frame: pl.DataFrame) -> pl.DataFrame:
    """Add an egfr column computed from scr, age, and sex.

    Args:
        frame: ml.cardiovascular_training rows.

    Returns:
        The frame with egfr. Null where an input is missing.
    """
    inputs = [frame[name].cast(pl.Float64).to_numpy() for name in EGFR_INPUTS]
    egfr = pl.Series("egfr", egfr_ckd_epi_2021_array(*inputs), nan_to_null=True)
    return frame.with_columns(egfr)


def add_egfr(frame: pd.DataFrame) -> pd.DataFrame:
    """Pandas variant of with_egfr, kept for the quantile_prediction notebook.

    Args:
        frame: ml.cardiovascular_training rows as a pandas frame.

    Returns:
        A copy of the frame with egfr. NaN where an input is missing.
    """
    result = frame.copy()
    result["egfr"] = egfr_ckd_epi_2021_array(
        *(pd.to_numeric(frame[name]).to_numpy(dtype=np.float64) for name in EGFR_INPUTS)
    )
    return result
