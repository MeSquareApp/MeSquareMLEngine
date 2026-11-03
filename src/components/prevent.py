"""AHA PREVENT base model, 10-year risk of total CVD, ASCVD, and heart failure.

Port of the official AHA PREVENT package v1.1.0 (github.com/AHA-DS-Analytics/PREVENT,
`pred_risk_base` in R and `aha_prevent_base` in Stata). tests/test_prevent.py checks these
coefficients against the package's `prevent_beta10_2024.dta` whenever a PREVENT checkout is
available. Inputs are vectorized so posterior samples can go through in one call.

Sex uses NHANES codes here (1 = male, 2 = female). The official package uses 0 = male,
1 = female.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# mmol_conversion() in the official package.
MMOL_L_PER_MG_DL = 0.02586
SEX_MALE = 1
SEX_FEMALE = 2
AGE_RANGE = (30, 79)
# Validated input ranges from aha_prevent_base.ado. Outside them, Stata returns missing.
VALIDATED_RANGES = {
    "total_chol": (130.0, 320.0),
    "hdl": (20.0, 100.0),
    "sbp": (90.0, 200.0),
    "bmi": (18.5, 39.99),
}
OUTCOMES = ("total_cvd", "ascvd", "heart_failure")

# Order: const, cage, cnhdl, chdl, csbp_lo, csbp_hi, dm, smoking, cegfr_lo, cegfr_hi, bptx, statin,
# bptx*csbp_hi, statin*cnhdl, cage*cnhdl, cage*chdl, cage*csbp_hi, cage*dm, cage*smoking,
# cage*cegfr_lo
_LIPID_TERMS = {
    ("total_cvd", SEX_FEMALE): (
        -3.307728, 0.7939329, 0.0305239, -0.1606857, -0.2394003, 0.360078, 0.8667604,
        0.5360739, 0.6045917, 0.0433769, 0.3151672, -0.1477655, -0.0663612, 0.1197879,
        -0.0819715, 0.0306769, -0.0946348, -0.27057, -0.078715, -0.1637806,
    ),
    ("total_cvd", SEX_MALE): (
        -3.031168, 0.7688528, 0.0736174, -0.0954431, -0.4347345, 0.3362658, 0.7692857,
        0.4386871, 0.5378979, 0.0164827, 0.288879, -0.1337349, -0.0475924, 0.150273,
        -0.0517874, 0.0191169, -0.1049477, -0.2251948, -0.0895067, -0.1543702,
    ),
    ("ascvd", SEX_FEMALE): (
        -3.819975, 0.719883, 0.1176967, -0.151185, -0.0835358, 0.3592852, 0.8348585,
        0.4831078, 0.4864619, 0.0397779, 0.2265309, -0.0592374, -0.0395762, 0.0844423,
        -0.0567839, 0.0325692, -0.1035985, -0.2417542, -0.0791142, -0.1671492,
    ),
    ("ascvd", SEX_MALE): (
        -3.500655, 0.7099847, 0.1658663, -0.1144285, -0.2837212, 0.3239977, 0.7189597,
        0.3956973, 0.3690075, 0.0203619, 0.2036522, -0.0865581, -0.0322916, 0.114563,
        -0.0300005, 0.0232747, -0.0927024, -0.2018525, -0.0970527, -0.1217081,
    ),
}

# Order: const, cage, csbp_lo, csbp_hi, dm, smoking, cbmi_lo, cbmi_hi, cegfr_lo, cegfr_hi, bptx,
# bptx*csbp_hi, cage*csbp_hi, cage*dm, cage*smoking, cage*cbmi_hi, cage*cegfr_lo
_HF_TERMS = {
    SEX_FEMALE: (
        -4.310409, 0.8998235, -0.4559771, 0.3576505, 1.038346, 0.583916, -0.0072294,
        0.2997706, 0.7451638, 0.0557087, 0.3534442, -0.0981511, -0.0946663, -0.3581041,
        -0.1159453, -0.003878, -0.1884289,
    ),
    SEX_MALE: (
        -3.946391, 0.8972642, -0.6811466, 0.3634461, 0.923776, 0.5023736, -0.0485841,
        0.3726929, 0.6926917, 0.0251827, 0.2980922, -0.0497731, -0.1289201, -0.3040924,
        -0.1401688, 0.0068126, -0.1797778,
    ),
}


@dataclass(frozen=True)
class PreventInputs:
    """Cholesterol in mg/dL, SBP in mmHg, BMI in kg/m2, eGFR in mL/min/1.73m2.

    Flags are 0/1. Any field may be an array of samples.
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


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def prevent_risk(inputs: PreventInputs) -> dict[str, np.ndarray]:
    """10-year risks as fractions. Raises ValueError outside the validated age range."""
    if not AGE_RANGE[0] <= inputs.age <= AGE_RANGE[1]:
        raise ValueError(f"PREVENT is valid for ages {AGE_RANGE[0]}-{AGE_RANGE[1]}")
    if inputs.sex not in (SEX_MALE, SEX_FEMALE):
        raise ValueError(f"Unsupported sex code: {inputs.sex!r}")

    total_chol = np.asarray(inputs.total_chol, dtype=float)
    hdl = np.asarray(inputs.hdl, dtype=float)
    sbp = np.asarray(inputs.sbp, dtype=float)
    bmi = np.asarray(inputs.bmi, dtype=float)
    egfr = np.asarray(inputs.egfr, dtype=float)
    dm = np.asarray(inputs.diabetes, dtype=float)
    smk = float(inputs.smoking)
    bptx = float(inputs.bp_treated)
    statin = float(inputs.statin)

    cage = (inputs.age - 55.0) / 10.0
    cnhdl = (total_chol - hdl) * MMOL_L_PER_MG_DL - 3.5
    chdl = (hdl * MMOL_L_PER_MG_DL - 1.3) / 0.3
    csbp_lo = (np.minimum(sbp, 110.0) - 110.0) / 20.0
    csbp_hi = (np.maximum(sbp, 110.0) - 130.0) / 20.0
    cegfr_lo = (np.minimum(egfr, 60.0) - 60.0) / -15.0
    cegfr_hi = (np.maximum(egfr, 60.0) - 90.0) / -15.0
    cbmi_lo = (np.minimum(bmi, 30.0) - 25.0) / 5.0
    cbmi_hi = (np.maximum(bmi, 30.0) - 30.0) / 5.0

    lipid_x = (
        1.0, cage, cnhdl, chdl, csbp_lo, csbp_hi, dm, smk, cegfr_lo, cegfr_hi, bptx, statin,
        bptx * csbp_hi, statin * cnhdl, cage * cnhdl, cage * chdl, cage * csbp_hi, cage * dm,
        cage * smk, cage * cegfr_lo,
    )
    hf_x = (
        1.0, cage, csbp_lo, csbp_hi, dm, smk, cbmi_lo, cbmi_hi, cegfr_lo, cegfr_hi, bptx,
        bptx * csbp_hi, cage * csbp_hi, cage * dm, cage * smk, cage * cbmi_hi, cage * cegfr_lo,
    )

    def linear(coefs: tuple[float, ...], terms: tuple) -> np.ndarray:
        return sum(c * x for c, x in zip(coefs, terms, strict=True))

    return {
        "total_cvd": _logistic(linear(_LIPID_TERMS[("total_cvd", inputs.sex)], lipid_x)),
        "ascvd": _logistic(linear(_LIPID_TERMS[("ascvd", inputs.sex)], lipid_x)),
        "heart_failure": _logistic(linear(_HF_TERMS[inputs.sex], hf_x)),
    }


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
