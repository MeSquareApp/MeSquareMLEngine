"""Response models shared by /v1/predict and /v1/twin/state."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel


class ChangeOut(BaseModel):
    """Latest lab vs the previous one against the 95% reference change value."""

    previous_lab_id: str
    previous_value: float
    previous_measured_at: date
    percent_change: float
    rcv_low_pct: float
    rcv_high_pct: float
    significant: bool


class PredictionOut(BaseModel):
    """The served band of one analyte and what it is based on."""

    analyte: str
    basis: Literal["population", "lab", "population_wearable", "lab_wearable"]
    basis_label: str
    p10: float
    p50: float
    p90: float
    unit: str
    as_of: date
    source_lab_id: str | None = None
    lab_measured_at: date | None = None
    change: ChangeOut | None = None


class InputUsedOut(BaseModel):
    """One PREVENT input and whether it was measured, reported or predicted."""

    value: float
    source: Literal["measured", "reported", "predicted"]


class RiskResultOut(BaseModel):
    """Risk for one outcome and horizon, in percent."""

    outcome: Literal["total_cvd", "ascvd", "heart_failure"]
    horizon_years: int
    risk_pct: float
    risk_low: float
    risk_high: float
    category: Literal["low", "borderline", "intermediate", "high"] | None = None


class RiskOut(BaseModel):
    """PREVENT risk response.

    risk_basis is "measured" when every lab-type input (non-HDL, HDL, SBP, eGFR)
    was measured, "estimated" when none was, else "mixed". The app should label a
    risk that is not "measured" as an estimate. out_of_range lists inputs outside
    PREVENT's validated range, which were clipped to it.
    """

    status: Literal["generated", "not_generated"]
    risk_basis: Literal["measured", "mixed", "estimated"] | None = None
    inputs_used: dict[str, InputUsedOut] = {}
    results: list[RiskResultOut] = []
    missing_inputs: list[str] = []
    reason: str | None = None
    out_of_range: list[str] = []


class RejectedRecordOut(BaseModel):
    """A record that was not used, and why."""

    id: str | None
    biomarker_abbr: str | None
    reason: str


class WindowOut(BaseModel):
    """One wearable metric over one window ending at as_of."""

    metric: str
    days: int
    coverage: float
    sufficient: bool
    mean: float | None = None
    sd: float | None = None
    slope_per_day: float | None = None


class LabPromptOut(BaseModel):
    """When the app should ask the user for their next lab."""

    due_on: date
    due: bool
