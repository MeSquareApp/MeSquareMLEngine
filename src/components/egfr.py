"""CKD-EPI 2009 estimated glomerular filtration rate, including race."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

FloatArray = npt.NDArray[np.float64]

SEX_MALE = 1
SEX_FEMALE = 2
# NHANES RIDRETH3: 4 = Non-Hispanic Black. The 2009 equation uses one race factor.
RACE_NON_HISPANIC_BLACK = 4
EGFR_INPUTS = ("scr", "age", "sex", "race_eth")


def egfr_ckd_epi_2009_array(
    scr: npt.ArrayLike, age: npt.ArrayLike, sex: npt.ArrayLike, race_eth: npt.ArrayLike
) -> FloatArray:
    """Vectorized CKD-EPI 2009 eGFR in mL/min/1.73m2.

    Equation: Levey et al., Ann Intern Med 2009 (CKD-EPI creatinine).

    Args:
        scr: Serum creatinine in mg/dL.
        age: Age in years.
        sex: NHANES RIAGENDR codes, 1 = male, 2 = female.
        race_eth: NHANES RIDRETH3 codes. Code 4 applies the Black race factor.

    Returns:
        eGFR per row. NaN where any input is missing or sex is not 1 or 2.
    """
    scr_values = np.asarray(scr, dtype=np.float64)
    sex_values = np.asarray(sex, dtype=np.float64)
    female = sex_values == SEX_FEMALE
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.329, -0.411)
    ratio = scr_values / kappa
    egfr = (
        141.0
        * np.minimum(ratio, 1.0) ** alpha
        * np.maximum(ratio, 1.0) ** -1.209
        * 0.993 ** np.asarray(age, dtype=np.float64)
        * np.where(female, 1.018, 1.0)
        * np.where(np.asarray(race_eth) == RACE_NON_HISPANIC_BLACK, 1.159, 1.0)
    )
    valid_sex = female | (sex_values == SEX_MALE)
    return np.where(valid_sex, egfr, np.nan)


def egfr_ckd_epi_2009(scr: float, age: float, sex: int, race_eth: int) -> float:
    """CKD-EPI 2009 eGFR in mL/min/1.73m2 for one person.

    Args:
        scr: Serum creatinine in mg/dL.
        age: Age in years.
        sex: NHANES RIAGENDR code, 1 = male, 2 = female.
        race_eth: NHANES RIDRETH3 code. Code 4 applies the Black race factor.

    Returns:
        eGFR.

    Raises:
        ValueError: If sex is not 1 or 2.
    """
    if sex not in (SEX_MALE, SEX_FEMALE):
        raise ValueError(f"Unsupported sex code: {sex!r}")
    return float(egfr_ckd_epi_2009_array(scr, age, sex, race_eth))


def with_egfr(frame: pl.DataFrame) -> pl.DataFrame:
    """Add an egfr column computed from scr, age, sex, and race_eth.

    Args:
        frame: ml.cardiovascular_training rows.

    Returns:
        The frame with egfr. Null where an input is missing.
    """
    inputs = [frame[name].cast(pl.Float64).to_numpy() for name in EGFR_INPUTS]
    egfr = pl.Series("egfr", egfr_ckd_epi_2009_array(*inputs), nan_to_null=True)
    return frame.with_columns(egfr)


def add_egfr(frame: pd.DataFrame) -> pd.DataFrame:
    """Pandas variant of with_egfr, kept for the quantile_prediction notebook.

    Args:
        frame: ml.cardiovascular_training rows as a pandas frame.

    Returns:
        A copy of the frame with egfr. NaN where an input is missing.
    """
    result = frame.copy()
    result["egfr"] = egfr_ckd_epi_2009_array(
        *(pd.to_numeric(frame[name]).to_numpy(dtype=np.float64) for name in EGFR_INPUTS)
    )
    return result
