"""CKD-EPI 2009 estimated glomerular filtration rate, including race."""

from __future__ import annotations

import math

import pandas as pd

SEX_MALE = 1
SEX_FEMALE = 2
# NHANES RIDRETH3: 4 = Non-Hispanic Black. The 2009 equation uses one race factor.
RACE_NON_HISPANIC_BLACK = 4


def egfr_ckd_epi_2009(scr: float, age: float, sex: int, race_eth: int) -> float:
    """Return eGFR in mL/min/1.73m2 from serum creatinine (mg/dL), age, sex, and race.

    Sex codes match NHANES RIAGENDR: 1 = male, 2 = female.
    Race codes match NHANES RIDRETH3. Code 4 applies the Black race factor.
    Equation: Levey et al., Ann Intern Med 2009 (CKD-EPI creatinine).
    """
    if sex == SEX_FEMALE:
        kappa = 0.7
        alpha = -0.329
        female_factor = 1.018
    elif sex == SEX_MALE:
        kappa = 0.9
        alpha = -0.411
        female_factor = 1.0
    else:
        raise ValueError(f"Unsupported sex code: {sex!r}")

    race_factor = 1.159 if race_eth == RACE_NON_HISPANIC_BLACK else 1.0
    ratio = scr / kappa
    return (
        141.0
        * (min(ratio, 1.0) ** alpha)
        * (max(ratio, 1.0) ** -1.209)
        * (0.993**age)
        * female_factor
        * race_factor
    )


def add_egfr(frame: pd.DataFrame) -> pd.DataFrame:
    """Add egfr from scr, age, sex, and race_eth in ml.cardiovascular_training."""
    result = frame.copy()

    def _row_egfr(row: pd.Series) -> float | None:
        scr = row["scr"]
        age = row["age"]
        sex = row["sex"]
        race_eth = row["race_eth"]
        if pd.isna(scr) or pd.isna(age) or pd.isna(sex) or pd.isna(race_eth):
            return None
        value = egfr_ckd_epi_2009(float(scr), float(age), int(sex), int(race_eth))
        if math.isnan(value):
            return None
        return value

    result["egfr"] = result.apply(_row_egfr, axis=1)
    return result
