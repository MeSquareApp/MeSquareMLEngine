"""Request and response bodies for POST /v1/twin/state."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from src.schemas.outputs import (
    LabPromptOut,
    PredictionOut,
    RejectedRecordOut,
    RiskOut,
    WindowOut,
)
from src.schemas.prediction import PatientFeatures


class Profile(PatientFeatures):
    """Prediction features plus treatment changes that reset a lab anchor."""

    statin_started_at: date | None = None
    bp_meds_started_at: date | None = None
    pregnant: bool = False


class RecordIn(BaseModel):
    """One extracted lab result or wearable daily summary.

    Every field is optional here so that one bad record is rejected with a reason
    in rejected_records instead of failing the whole request.
    """

    id: str | None = None
    biomarker_abbr: str | None = None
    value: float | None = None
    unit: str | None = None
    measurement_method: str | None = None
    measured_at: datetime | None = None


class TwinStateRequest(BaseModel):
    """Body of POST /v1/twin/state: the user's profile and every record."""

    user_id: str | None = None
    profile: Profile
    records: list[RecordIn] = Field(default_factory=list)
    as_of: date | None = None


class TwinStateResponse(BaseModel):
    """Personalized bands, risk, and traceability for one user."""

    as_of: date
    predictions: list[PredictionOut]
    risk: RiskOut
    rejected_records: list[RejectedRecordOut]
    wearable_windows: dict[str, WindowOut]
    lab_prompt: LabPromptOut
    model_versions: dict[str, str]
    input_hash: str
