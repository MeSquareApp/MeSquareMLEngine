"""Request and response bodies for POST /v1/twin/predict."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.prediction import PatientFeatures

LabBiomarker = Literal["non_hdl", "hdl", "total_chol", "hba1c", "egfr", "scr", "sbp"]


class Profile(PatientFeatures):
    """Same fields and labels as /v1/predict, plus height for BMI in PREVENT."""

    height_cm: float | None = Field(default=None, gt=0)


class LabResult(BaseModel):
    date: date
    biomarker: LabBiomarker
    value: float = Field(gt=0)


class WearableDay(BaseModel):
    """End-of-day aggregates from Open Wearables. Any field may be missing."""

    date: date
    steps: float | None = None
    resting_hr: float | None = None
    hrv_rmssd: float | None = None
    sleep_h: float | None = None
    mvpa_min: float | None = None
    weight_kg: float | None = None


class TwinRequest(BaseModel):
    profile: Profile
    labs: list[LabResult] = []
    wearable_days: list[WearableDay] = []
    as_of: date | None = None
    horizon_days: int = Field(default=90, ge=0, le=365)
    include_trajectory: bool = True


class Interval(BaseModel):
    q10: float
    q50: float
    q90: float


class TrajectoryPoint(Interval):
    date: date
    basis: Literal["prior", "estimated", "measured", "anchored"]
    confidence: Literal["low", "medium", "high"]
    forecast: bool


class InnovationOut(BaseModel):
    date: date
    observed: float
    predicted: float
    z: float


class BiomarkerOut(BaseModel):
    unit: str
    prior: Interval
    current: TrajectoryPoint
    forecast: TrajectoryPoint
    change_vs_prior: float
    wearable_effect: float
    innovations: list[InnovationOut]
    trajectory: list[TrajectoryPoint] | None = None


class WearableOut(BaseModel):
    first_date: date | None
    baseline_end: date | None
    baseline_ready: bool
    baseline: dict[str, float | None]
    latest_delta: dict[str, float | None]


class RiskOut(BaseModel):
    median: float
    q10: float
    q90: float


class TwinResponse(BaseModel):
    as_of: date
    biomarkers: dict[str, BiomarkerOut]
    wearable: WearableOut
    # Keys: population_prior, current, forecast. Each maps outcome -> 10-year risk.
    prevent: dict[str, dict[str, RiskOut]]
    prevent_note: str | None = None
