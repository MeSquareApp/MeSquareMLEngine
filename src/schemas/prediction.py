"""Request and response schemas for quantile biomarker prediction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_serializer

from src.components.model_config import MAX_PREDICT_BATCH
from src.constants import Ethnicity, Sex
from src.schemas.outputs import PredictionOut, RiskOut

SmokingStatus = Literal["never", "former", "current"]
DiabetesDx = Literal["no", "yes", "borderline"]
BinaryFlag = Literal[0, 1]


class PatientFeatures(BaseModel):
    """Patient features. sex and race_eth take NHANES labels and dump as codes."""

    sex: Sex
    age: float = Field(ge=20, le=79)
    race_eth: Ethnicity
    height_cm: float | None = Field(default=None, gt=0)
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
        """Dump sex as its NHANES RIAGENDR code."""
        return value.code

    @field_serializer("race_eth")
    def _serialize_race_eth(self, value: Ethnicity) -> int:
        """Dump race_eth as its NHANES RIDRETH3 code."""
        return value.code


class PredictRequest(BaseModel):
    """Body of POST /v1/predict."""

    patients: list[PatientFeatures] = Field(min_length=1, max_length=MAX_PREDICT_BATCH)


class PatientPrediction(BaseModel):
    """Population bands and an estimated risk for one patient."""

    predictions: list[PredictionOut]
    risk: RiskOut
    input_hash: str


class PredictResponse(BaseModel):
    """Response of POST /v1/predict."""

    model_versions: dict[str, str]
    results: list[PatientPrediction]
