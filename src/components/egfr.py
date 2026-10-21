"""CKD-EPI 2021 (race-free) estimated glomerular filtration rate."""

from __future__ import annotations

import math

import pandas as pd

SEX_MALE = 1
SEX_FEMALE = 2


def egfr_ckd_epi_2021(scr: float, age: float, sex: int) -> float:
    """Return eGFR in mL/min/1.73m2 from serum creatinine (mg/dL), age, and NHANES sex.

    Sex codes match NHANES RIAGENDR: 1 = male, 2 = female.
    Equation: Inker et al., N Engl J Med 2021 (race-free CKD-EPI creatinine).
    """
    if sex == SEX_FEMALE:
        kappa = 0.7
        alpha = -0.241
        female_factor = 1.012
    elif sex == SEX_MALE:
        kappa = 0.9
        alpha = -0.302
        female_factor = 1.0
    else:
        raise ValueError(f"Unsupported sex code: {sex!r}")

    ratio = scr / kappa
    return (
        142.0
        * (min(ratio, 1.0) ** alpha)
        * (max(ratio, 1.0) ** -1.200)
        * (0.9938**age)
        * female_factor
    )


def add_y_egfr(frame: pd.DataFrame) -> pd.DataFrame:
    """Add y_egfr from scr, age, and sex columns produced by ml.cardiovascular_training."""
    result = frame.copy()

    def _row_egfr(row: pd.Series) -> float | None:
        scr = row["scr"]
        age = row["age"]
        sex = row["sex"]
        if pd.isna(scr) or pd.isna(age) or pd.isna(sex):
            return None
        value = egfr_ckd_epi_2021(float(scr), float(age), int(sex))
        if math.isnan(value):
            return None
        return value

    result["y_egfr"] = result.apply(_row_egfr, axis=1)
    return result
