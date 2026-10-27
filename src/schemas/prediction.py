"""Request and response schemas for quantile biomarker prediction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_serializer

from src.components.model_config import MAX_PREDICT_BATCH, TARGETS
from src.constants import Ethnicity, Sex

SmokingStatus = Literal["never", "former", "current"]
DiabetesDx = Literal["no", "yes", "borderline"]
BinaryFlag = Literal[0, 1]


class PatientFeatures(BaseModel):
    sex: Sex
    age: float = Field(ge=20, le=79)
    race_eth: Ethnicity
    weight_kg: float | None = Field(default=None, gt=0)
    waist_cm: float | None = Field(default=None, gt=0)
    smoking: SmokingStatus | None = None
    told_high_bp: BinaryFlag | None = None
    bp_meds: BinaryFlag | None = None
    told_high_chol: BinaryFlag | None = None
    chol_meds: BinaryFlag | None = None
    diabetes_dx: DiabetesDx | None = None
    prediabetes_dx: BinaryFlag | None = None
    insulin: BinaryFlag | None = None
    diabetes_pills: BinaryFlag | None = None

    @field_serializer("sex")
    def _serialize_sex(self, value: Sex) -> int:
        return value.code

    @field_serializer("race_eth")
    def _serialize_race_eth(self, value: Ethnicity) -> int:
        return value.code


class PredictRequest(BaseModel):
    patients: list[PatientFeatures] = Field(min_length=1, max_length=MAX_PREDICT_BATCH)


class QuantilePrediction(BaseModel):
    p10: float
    p50: float
    p90: float
    unit: str


class PatientPrediction(BaseModel):
    predictions: dict[str, QuantilePrediction]


class PredictResponse(BaseModel):
    model_version: str
    results: list[PatientPrediction]


# Ensure OpenAPI documents the expected biomarker keys.
assert set(TARGETS) == {"non_hdl", "hdl", "hba1c", "egfr"}
