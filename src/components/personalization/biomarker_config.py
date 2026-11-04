"""Within-person variation of each modelled analyte. Tune here, nowhere else.

measurement_cv combines the analytical CV and the within-person biological CV of one
measurement (EFLM Biological Variation Database: total and HDL cholesterol about
5-6%, HbA1c about 1.5%, creatinine about 4.5%, plus typical analytical CVs; home BP
protocol day-to-day CV about 6%). annual_drift_cv is slow within-person change over a
year with no change in treatment. Both are applied on the log scale, because every
analyte is right-skewed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BiomarkerSpec:
    """Variation settings for one analyte.

    Attributes:
        name: Analyte name, matching the prior model's target.
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
        BiomarkerSpec("non_hdl", "mg/dL", 0.07, 0.08, (20.0, 450.0)),
        BiomarkerSpec("hdl", "mg/dL", 0.065, 0.06, (5.0, 200.0)),
        BiomarkerSpec("hba1c", "%", 0.025, 0.04, (3.0, 20.0)),
        BiomarkerSpec("egfr", "mL/min/1.73m2", 0.06, 0.04, (2.0, 200.0)),
        BiomarkerSpec("sbp", "mmHg", 0.06, 0.05, (60.0, 260.0)),
    )
}

# Analytes whose lab anchor is reset when a treatment that changes them starts.
LIPID_ANALYTES = frozenset({"non_hdl", "hdl"})
BP_ANALYTES = frozenset({"sbp"})
