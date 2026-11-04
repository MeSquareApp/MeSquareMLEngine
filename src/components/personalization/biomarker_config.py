"""Within-person variation of each modelled biomarker. Tune here, nowhere else.

measurement_cv is the total CV of one measurement, sqrt(CV_I^2 + CV_A^2):

- CV_I: within-person biological CV, EFLM Biological Variation Database meta-analysis
  median (https://biologicalvariation.eu, estimates updated 2026-08-10, read
  2026-11-04): total cholesterol 5.4% (95% CI 4.9-6.6), HDL 5.7% (4.7-7.8), LDL 7.6%
  (7.0-8.4), creatinine 4.4% (2.3-4.7), HbA1c NGSP 1.2% (0.2-7.0). No entry for
  non-HDL.
- CV_A: EFLM desirable analytical imprecision, 0.5 * CV_I (Sandberg et al., Clin Chem
  Lab Med 2024;62:1483-9), until the partner lab's QC CV is known.

That gives HDL 6.4% (used: 6.5%), HbA1c 1.3%, eGFR 5.9% (1.2 x creatinine 4.9%,
CKD-EPI 2021 exponent; used: 6.0%) and non-HDL 8.5% (LDL proxy; 8.8% propagating total
and HDL). Home systolic BP: day-to-day CV of one morning reading 5.4% (Okada et al.,
Nihon Jinzo Gakkai Shi 2008;50:588-96; used: 6.0%).

annual_drift_cv is slow within-person change over a year with no change in
treatment. Lipids: total cholesterol drift about 3.5%/year (Glasziou et al., Ann Intern
Med 2008;148:656-61), so 8%/6% is cautious. HbA1c, eGFR and SBP drift are design
choices with no published source.

Both are applied on the log scale, because every biomarker is right-skewed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BiomarkerSpec:
    """Variation settings for one biomarker.

    Attributes:
        name: Biomarker name, matching the prior model's target.
        unit: Canonical unit.
        measurement_cv: Analytical plus within-person CV of a single measurement.
        annual_drift_cv: Within-person drift over a year with no treatment change.
        clamp: Plausible range for reported quantiles.
    """

    name: str
    unit: str
    measurement_cv: float
    annual_drift_cv: float
    clamp: tuple[float, float]


BIOMARKERS: dict[str, BiomarkerSpec] = {
    spec.name: spec
    for spec in (
        BiomarkerSpec("non_hdl", "mg/dL", 0.085, 0.08, (20.0, 450.0)),
        BiomarkerSpec("hdl", "mg/dL", 0.065, 0.06, (5.0, 200.0)),
        BiomarkerSpec("hba1c", "%", 0.013, 0.04, (3.0, 20.0)),
        BiomarkerSpec("egfr", "mL/min/1.73m2", 0.06, 0.04, (2.0, 200.0)),
        BiomarkerSpec("sbp", "mmHg", 0.06, 0.05, (60.0, 260.0)),
    )
}

# Biomarkers whose lab anchor is reset when a treatment that changes them starts.
LIPID_BIOMARKERS = frozenset({"non_hdl", "hdl"})
BP_BIOMARKERS = frozenset({"sbp"})
