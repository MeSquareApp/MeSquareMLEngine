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
    """One lab value. total_chol and scr are converted to non_hdl and eGFR."""

    date: date
    biomarker: LabBiomarker
    value: float = Field(gt=0)


class WearableDay(BaseModel):
    """End-of-day aggregates from Open Wearables. Any field may be missing.

    wear_hours lets the engine drop days the device was barely worn. Without it,
    days under 1000 steps are treated as not worn.
    """

    date: date
    wear_hours: float | None = Field(default=None, ge=0, le=24)
    steps: float | None = None
    resting_hr: float | None = None
    hrv_rmssd: float | None = None
    sleep_h: float | None = None
    mvpa_min: float | None = None
    weight_kg: float | None = None


class TwinRequest(BaseModel):
    """Body of POST /v1/twin/predict: the user's full history."""

    profile: Profile
    labs: list[LabResult] = Field(default_factory=list)
    wearable_days: list[WearableDay] = Field(default_factory=list)
    as_of: date | None = None
    horizon_days: int = Field(default=90, ge=0, le=365)
    include_trajectory: bool = True


class Interval(BaseModel):
    """Natural-unit 10th, 50th, and 90th percentiles."""

    q10: float
    q50: float
    q90: float


class TrajectoryPoint(Interval):
    """One day of a biomarker trajectory and where its estimate comes from."""

    date: date
    basis: Literal["prior", "estimated", "measured", "anchored"]
    confidence: Literal["low", "medium", "high"]
    forecast: bool


class InnovationOut(BaseModel):
    """How a lab compared with the prediction made just before it."""

    date: date
    observed: float
    predicted: float
    z: float


class BiomarkerOut(BaseModel):
    """Prior, current, and forecast view of one biomarker."""

    unit: str
    prior: Interval
    current: TrajectoryPoint
    forecast: TrajectoryPoint
    change_vs_prior: float
    wearable_effect: float
    innovations: list[InnovationOut]
    trajectory: list[TrajectoryPoint] | None = None


class WearableOut(BaseModel):
    """The user's wearable baseline and latest change against it."""

    first_date: date | None
    baseline_end: date | None
    baseline_ready: bool
    baseline: dict[str, float | None]
    latest_delta: dict[str, float | None]
    valid_days_last_30: int
    last_valid_date: date | None
    status: Literal["no_data", "building_baseline", "ok", "sparse", "stale"]


class RiskOut(BaseModel):
    """10-year risk as a fraction: median and 80% interval."""

    median: float
    q10: float
    q90: float


class TwinResponse(BaseModel):
    """Twin output. prevent keys: population_prior, current, forecast."""

    as_of: date
    biomarkers: dict[str, BiomarkerOut]
    wearable: WearableOut
    prevent: dict[str, dict[str, RiskOut]]
    prevent_note: str | None = None
