import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.components.prevent import SEX_FEMALE, SEX_MALE, PreventInputs, prevent_risk

# preventr reference: age 50, TC 200, HDL 45, SBP 160 treated, diabetes, non-smoker,
# eGFR 90, BMI 35, no statin. Published 10-year base-model risks.
REFERENCE = {
    2: {"total_cvd": 0.147, "ascvd": 0.092, "heart_failure": 0.081},
    1: {"total_cvd": 0.163, "ascvd": 0.102, "heart_failure": 0.106},
}


def reference_inputs(sex: int, **overrides) -> PreventInputs:
    values = dict(
        age=50, sex=sex, total_chol=200, hdl=45, sbp=160, bmi=35, egfr=90,
        diabetes=1, smoking=0, bp_treated=1, statin=0,
    )
    values.update(overrides)
    return PreventInputs(**values)


@pytest.mark.parametrize("sex", [1, 2])
def test_matches_reference(sex):
    risk = prevent_risk(reference_inputs(sex))
    for outcome, expected in REFERENCE[sex].items():
        assert round(float(risk[outcome]), 3) == expected


def test_vectorized_samples():
    risk = prevent_risk(reference_inputs(2, sbp=np.array([120.0, 160.0])))
    assert risk["total_cvd"].shape == (2,)
    assert risk["total_cvd"][0] < risk["total_cvd"][1]


def test_age_out_of_range():
    with pytest.raises(ValueError):
        prevent_risk(reference_inputs(1, age=25))


def _official_dir() -> Path | None:
    """The AHA PREVENT checkout: PREVENT_REPO_DIR, or a sibling PREVENT folder up the tree."""
    candidates = [os.environ.get("PREVENT_REPO_DIR")]
    candidates += [str(parent / "PREVENT") for parent in Path(__file__).resolve().parents]
    for candidate in candidates:
        if candidate and (Path(candidate) / "a" / "prevent_beta10_2024.dta").exists():
            return Path(candidate)
    return None


OFFICIAL = _official_dir()


def _official_risk(row: pd.Series, betas: pd.DataFrame, outcome: str) -> float:
    """Build the 10-year linear predictor the way aha_prevent_base.ado does: same term order,
    official beta table, official sex coding (0 = male, 1 = female)."""
    age = (row.age - 55) / 10
    sbp1, sbp2 = min((row.sbp - 110) / 20, 0), max((row.sbp - 110) / 20, 0) - 1
    nhdl = row.tc * 0.02586 - row.hdl * 0.02586 - 3.5
    hdl = (row.hdl * 0.02586 - 1.3) / 0.3
    bmi1, bmi2 = min((row.bmi - 25) / 5, 1), max((row.bmi - 25) / 5 - 1, 0)
    egfr1 = -min(row.egfr, 60) / 15 + 4
    egfr2 = -max(row.egfr - 60, 0) / 15 + 2
    if outcome == "hf":
        x = [age, sbp1, sbp2, row.dm, row.smoke, bmi1, bmi2, egfr1, egfr2, row.htnmed,
             sbp2 * row.htnmed, age * sbp2, age * row.dm, age * row.smoke, age * bmi2,
             age * egfr1, 1.0]
    else:
        x = [age, nhdl, hdl, sbp1, sbp2, row.dm, row.smoke, egfr1, egfr2, row.htnmed,
             row.statin, sbp2 * row.htnmed, nhdl * row.statin, age * nhdl, age * hdl,
             age * sbp2, age * row.dm, age * row.smoke, age * egfr1, 1.0]
    beta = betas[f"beta1{int(row.sex)}_{outcome}1"].iloc[: len(x)].to_numpy()
    xb = float(np.dot(beta, x))
    return np.exp(xb) / (1 + np.exp(xb))


@pytest.mark.skipif(OFFICIAL is None, reason="AHA PREVENT checkout not found")
def test_matches_official_package_on_its_test_data():
    betas = pd.read_stata(OFFICIAL / "a" / "prevent_beta10_2024.dta")
    cases = pd.read_stata(OFFICIAL / "a" / "prevent_test.dta")
    required = ["age", "sex", "tc", "hdl", "sbp", "bmi", "egfr", "dm", "smoke", "htnmed", "statin"]
    cases = cases.dropna(subset=required)
    cases = cases[
        cases.age.between(30, 79) & cases.tc.between(130, 320) & cases.hdl.between(20, 100)
        & cases.sbp.between(90, 200) & cases.bmi.between(18.5, 39.99)
    ]
    assert len(cases) > 0
    for row in cases.itertuples():
        ours = prevent_risk(
            PreventInputs(
                age=row.age, sex=SEX_MALE if row.sex == 0 else SEX_FEMALE, total_chol=row.tc,
                hdl=row.hdl, sbp=row.sbp, bmi=row.bmi, egfr=row.egfr, diabetes=row.dm,
                smoking=row.smoke, bp_treated=row.htnmed, statin=row.statin,
            )
        )
        for outcome, official_name in (("total_cvd", "cvd"), ("ascvd", "ascvd"), ("heart_failure", "hf")):
            assert float(ours[outcome]) == pytest.approx(
                _official_risk(row, betas, official_name), rel=1e-6
            )
