"""Integration tests against the installed official AHAprevent R package.

They skip when R or the package is missing. Unit tests with a mocked R call live in
test_prevent.py.
"""

import os
from pathlib import Path

import numpy as np
import polars as pl
import pyreadstat
import pytest

from src.components.prevent import (
    SEX_FEMALE,
    SEX_MALE,
    PreventInputs,
    prevent_available,
    prevent_risk,
)

pytestmark = pytest.mark.skipif(
    not prevent_available(), reason="R or the AHAprevent package is not installed"
)

# Published reference case from preventr, an independent implementation: age 50,
# TC 200, HDL 45, SBP 160 treated, diabetes, non-smoker, eGFR 90, BMI 35, no statin.
REFERENCE = {
    SEX_FEMALE: {"total_cvd": 0.147, "ascvd": 0.092, "heart_failure": 0.081},
    SEX_MALE: {"total_cvd": 0.163, "ascvd": 0.102, "heart_failure": 0.106},
}
MMOL_PER_MG = 0.02586
REQUIRED = ("age", "sex", "tc", "hdl", "sbp", "bmi", "egfr", "dm", "smoke", "htnmed")


def reference_inputs(sex: int) -> PreventInputs:
    return PreventInputs(
        age=50,
        sex=sex,
        total_chol=200,
        hdl=45,
        sbp=160,
        bmi=35,
        egfr=90,
        diabetes=1,
        smoking=0,
        bp_treated=1,
        statin=0,
    )


def _official_dir() -> Path | None:
    """PREVENT checkout: PREVENT_REPO_DIR, or a sibling PREVENT folder up the tree."""
    candidates = [os.environ.get("PREVENT_REPO_DIR")]
    candidates += [
        str(parent / "PREVENT") for parent in Path(__file__).resolve().parents
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "a" / "prevent_beta10_2024.dta").exists():
            return Path(candidate)
    return None


OFFICIAL = _official_dir()


def _read_dta(path: Path) -> pl.DataFrame:
    data, _ = pyreadstat.read_dta(str(path), output_format="dict")
    return pl.DataFrame(data)


def _beta_risk(row: dict[str, float], betas: pl.DataFrame, outcome: str) -> float:
    """10-year risk built from the beta table term by term, like aha_prevent_base.ado.

    Official sex coding: 0 = male, 1 = female.
    """
    age = (row["age"] - 55) / 10
    sbp1 = min((row["sbp"] - 110) / 20, 0)
    sbp2 = max((row["sbp"] - 110) / 20, 0) - 1
    nhdl = (row["tc"] - row["hdl"]) * MMOL_PER_MG - 3.5
    hdl = (row["hdl"] * MMOL_PER_MG - 1.3) / 0.3
    bmi1 = min((row["bmi"] - 25) / 5, 1)
    bmi2 = max((row["bmi"] - 25) / 5 - 1, 0)
    egfr1 = -min(row["egfr"], 60) / 15 + 4
    egfr2 = -max(row["egfr"] - 60, 0) / 15 + 2
    dm, smoke, bptx, statin = row["dm"], row["smoke"], row["htnmed"], row["statin"]
    if outcome == "hf":
        terms = [
            age, sbp1, sbp2, dm, smoke, bmi1, bmi2, egfr1, egfr2, bptx, sbp2 * bptx,
            age * sbp2, age * dm, age * smoke, age * bmi2, age * egfr1, 1.0,
        ]  # fmt: skip
    else:
        terms = [
            age, nhdl, hdl, sbp1, sbp2, dm, smoke, egfr1, egfr2, bptx, statin,
            sbp2 * bptx, nhdl * statin, age * nhdl, age * hdl, age * sbp2, age * dm,
            age * smoke, age * egfr1, 1.0,
        ]  # fmt: skip
    column = f"beta1{int(row['sex'])}_{outcome}1"
    beta = betas[column].head(len(terms)).to_numpy()
    linear = float(np.dot(beta, terms))
    return float(np.exp(linear) / (1 + np.exp(linear)))


@pytest.mark.parametrize("sex", [SEX_MALE, SEX_FEMALE])
def test_official_package_matches_published_reference(sex):
    # Arrange
    inputs = reference_inputs(sex)

    # Act
    risk = prevent_risk(inputs)

    # Assert
    for outcome, expected in REFERENCE[sex].items():
        assert round(float(risk[outcome][0]), 3) == expected


@pytest.mark.skipif(OFFICIAL is None, reason="AHA PREVENT checkout not found")
def test_official_package_matches_beta_table_on_its_test_data():
    # Arrange
    assert OFFICIAL is not None
    betas = _read_dta(OFFICIAL / "a" / "prevent_beta10_2024.dta")
    cases = (
        _read_dta(OFFICIAL / "a" / "prevent_test.dta")
        .drop_nulls([*REQUIRED, "statin"])
        .filter(
            pl.col("age").is_between(30, 79),
            pl.col("tc").is_between(130, 320),
            pl.col("hdl").is_between(20, 100),
            pl.col("sbp").is_between(90, 200),
            pl.col("bmi").is_between(18.5, 39.99),
        )
    )
    outcomes = {"total_cvd": "cvd", "ascvd": "ascvd", "heart_failure": "hf"}

    # Act
    results = [
        (
            row,
            prevent_risk(
                PreventInputs(
                    age=row["age"],
                    sex=SEX_MALE if row["sex"] == 0 else SEX_FEMALE,
                    total_chol=row["tc"],
                    hdl=row["hdl"],
                    sbp=row["sbp"],
                    bmi=row["bmi"],
                    egfr=row["egfr"],
                    diabetes=row["dm"],
                    smoking=row["smoke"],
                    bp_treated=row["htnmed"],
                    statin=row["statin"],
                )
            ),
        )
        for row in cases.iter_rows(named=True)
    ]

    # Assert
    assert len(results) > 0
    for row, risk in results:
        for outcome, official_name in outcomes.items():
            expected = _beta_risk(row, betas, official_name)
            assert float(risk[outcome][0]) == pytest.approx(expected, rel=1e-6)
