"""Cardiovascular risk response: official PREVENT on the best available inputs.

Each PREVENT input is measured (a lab or reading), reported (the profile), or
predicted (the served p50 when no usable lab exists). risk_pct is PREVENT on those
values. risk_low and risk_high are PREVENT at the p10/p90 ends of every predicted
input, all combinations, so the range shows how much the estimate could move; with
only measured inputs the range collapses to the point. The method is deterministic
and needs one small call to the official package.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.components.prevent import (
    HORIZONS,
    OUTCOMES,
    VALIDATED_RANGES,
    PreventInputs,
    clip_to_validated,
    prevent_risk,
)

InputSource = Literal["measured", "reported", "predicted"]
RiskBasis = Literal["measured", "mixed", "estimated"]

LAB_INPUTS = ("non_hdl", "hdl", "sbp", "egfr")
HBA1C_DIABETES = 6.5
# SBP has a J-shaped effect below 110 mmHg, so 110 joins the grid when inside a band.
SBP_SPLINE_KNOT = 110.0
# AHA 10-year total CVD categories (PREVENT): upper bounds in percent.
CATEGORIES = ((5.0, "low"), (7.5, "borderline"), (20.0, "intermediate"))


@dataclass(frozen=True)
class InputEstimate:
    """One PREVENT input and where it came from.

    Attributes:
        value: Value used for the point estimate.
        source: "measured", "reported", or "predicted".
        band: (p10, p90) of a predicted value, None otherwise.
    """

    value: float
    source: InputSource
    band: tuple[float, float] | None = None


@dataclass(frozen=True)
class RiskProfile:
    """Fixed PREVENT inputs from the profile.

    Attributes:
        age: Age in years.
        sex: NHANES code, 1 = male, 2 = female.
        smoking: 1.0 if current smoker.
        bp_treated: 1.0 if on BP medication.
        statin: 1.0 if on a statin.
    """

    age: float
    sex: int
    smoking: float
    bp_treated: float
    statin: float


@dataclass(frozen=True)
class RiskResult:
    """Risk for one outcome and horizon.

    Attributes:
        outcome: "total_cvd", "ascvd", or "heart_failure".
        horizon_years: 10 or 30.
        risk_pct: Risk in percent from the point inputs.
        risk_low: Lowest risk across predicted inputs' p10/p90 ends, percent.
        risk_high: Highest risk across predicted inputs' p10/p90 ends, percent.
        category: AHA category for 10-year total CVD, else None.
    """

    outcome: str
    horizon_years: int
    risk_pct: float
    risk_low: float
    risk_high: float
    category: str | None


@dataclass(frozen=True)
class RiskAssessment:
    """The risk block of an API response.

    Attributes:
        status: "generated" or "not_generated".
        risk_basis: "measured", "mixed", or "estimated" when generated.
        inputs_used: Each input with its value and source.
        results: One entry per outcome and horizon.
        missing_inputs: Profile fields needed but absent.
        reason: Why no risk was generated, if not because of missing inputs.
        out_of_range: Inputs outside PREVENT's validated range (clipped to it).
    """

    status: Literal["generated", "not_generated"]
    risk_basis: RiskBasis | None = None
    inputs_used: dict[str, InputEstimate] = field(default_factory=dict)
    results: list[RiskResult] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    reason: str | None = None
    out_of_range: list[str] = field(default_factory=list)


def risk_category(risk_pct: float) -> str:
    """AHA category of a 10-year total CVD risk.

    Args:
        risk_pct: Risk in percent.

    Returns:
        "low" (<5), "borderline" (5-7.5), "intermediate" (7.5-20) or "high".
    """
    for upper, label in CATEGORIES:
        if risk_pct < upper:
            return label
    return "high"


def risk_basis(inputs: Mapping[str, InputEstimate]) -> RiskBasis:
    """Whether the lab-type inputs were measured, predicted, or both.

    Args:
        inputs: Inputs keyed by name. LAB_INPUTS are judged.

    Returns:
        "measured" when all are measured, "estimated" when none are, else "mixed".
    """
    measured = [inputs[name].source == "measured" for name in LAB_INPUTS]
    if all(measured):
        return "measured"
    return "mixed" if any(measured) else "estimated"


def _corner_values(name: str, estimate: InputEstimate) -> list[float]:
    """Band ends (and the SBP knot) of a predicted input, else its fixed value."""
    if estimate.source != "predicted" or estimate.band is None:
        return [estimate.value]
    low, high = estimate.band
    values = [low, high]
    if name == "sbp" and low < SBP_SPLINE_KNOT < high:
        values.append(SBP_SPLINE_KNOT)
    return values


def assess_risk(
    profile: RiskProfile, inputs: Mapping[str, InputEstimate]
) -> RiskAssessment:
    """Run official PREVENT on the point inputs and their p10/p90 combinations.

    Args:
        profile: Fixed profile inputs.
        inputs: "non_hdl", "hdl", "sbp", "egfr", "bmi", "hba1c" and "diabetes"
            estimates. Diabetes is the reported diagnosis (1.0 or 0.0); an HbA1c
            of 6.5% or more also counts as diabetes.

    Returns:
        A generated risk assessment.

    Raises:
        PreventUnavailableError: If R or the package is missing.
        KeyError: If an input is missing.
    """
    names = list(inputs)
    corners = itertools.product(*(_corner_values(n, inputs[n]) for n in names))
    rows = np.array([[inputs[name].value for name in names], *corners])
    column = {name: rows[:, index] for index, name in enumerate(names)}
    total_chol = column["non_hdl"] + column["hdl"]
    diabetes = np.maximum(column["diabetes"], column["hba1c"] >= HBA1C_DIABETES)
    point = {
        "total_chol": float(total_chol[0]),
        "hdl": inputs["hdl"].value,
        "sbp": inputs["sbp"].value,
        "bmi": inputs["bmi"].value,
    }
    risks = prevent_risk(
        PreventInputs(
            age=profile.age,
            sex=profile.sex,
            total_chol=clip_to_validated("total_chol", total_chol),
            hdl=clip_to_validated("hdl", column["hdl"]),
            sbp=clip_to_validated("sbp", column["sbp"]),
            bmi=clip_to_validated("bmi", column["bmi"]),
            egfr=column["egfr"],
            diabetes=diabetes.astype(np.float64),
            smoking=profile.smoking,
            bp_treated=profile.bp_treated,
            statin=profile.statin,
        )
    )
    results = []
    for outcome, horizon in itertools.product(OUTCOMES, HORIZONS):
        values = 100.0 * risks[outcome, horizon]
        if np.isnan(values[0]):
            continue
        results.append(
            RiskResult(
                outcome=outcome,
                horizon_years=horizon,
                risk_pct=float(values[0]),
                risk_low=float(np.nanmin(values)),
                risk_high=float(np.nanmax(values)),
                category=risk_category(float(values[0]))
                if (outcome, horizon) == ("total_cvd", 10)
                else None,
            )
        )
    out_of_range = [
        name
        for name, value in point.items()
        if not VALIDATED_RANGES[name][0] <= value <= VALIDATED_RANGES[name][1]
    ]
    return RiskAssessment(
        status="generated",
        risk_basis=risk_basis(inputs),
        inputs_used=dict(inputs),
        results=results,
        out_of_range=out_of_range,
    )
