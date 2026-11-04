"""Ingest extracted lab results and wearable daily summaries in one record shape.

Every record looks like::

    {"id": "<uuid>", "biomarker_abbr": "HDL", "value": 52, "unit": "mg/dL",
     "measurement_method": "blood_test_lab_report_upload",
     "measured_at": "2026-11-03T20:50:55.898Z"}

Records are mapped to a canonical biomarker, converted to its canonical unit, checked
against plausibility limits and the measurement method, or rejected with a reason
(and logged). The engine then derives non-HDL (total cholesterol minus HDL from the
same draw) and eGFR (CKD-EPI 2021 from creatinine) itself.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from src.components.biomarkers import (
    ALIAS_TO_BIOMARKER,
    BIOMARKER_CATALOG,
    HBA1C_NGSP_INTERCEPT,
    HBA1C_NGSP_SLOPE,
    KG_PER_LB,
    MEASURED_LAB_METHODS,
    MG_DL_PER_MMOL_L_CHOLESTEROL,
    UMOL_L_PER_MG_DL_CREATININE,
    WEARABLE_METHOD_PREFIXES,
    is_plausible,
    normalize_name,
)
from src.components.egfr import egfr_ckd_epi_2021
from src.utils.logger import get_logger

logger = get_logger(__name__)

RawRecord = Mapping[str, object]
REQUIRED_FIELDS = (
    "id",
    "biomarker_abbr",
    "value",
    "unit",
    "measurement_method",
    "measured_at",
)
CHOLESTEROL_BIOMARKERS = frozenset({"total_chol", "hdl", "non_hdl", "ldl"})

_IDENTITY: Callable[[float], float] = float
# Unit spellings are normalized with _normalize_unit before lookup.
UNIT_CONVERSIONS: dict[str, dict[str, Callable[[float], float]]] = {
    **{
        name: {
            "mg/dl": _IDENTITY,
            "mmol/l": lambda value: value * MG_DL_PER_MMOL_L_CHOLESTEROL,
        }
        for name in CHOLESTEROL_BIOMARKERS
    },
    "triglycerides": {"mg/dl": _IDENTITY},
    "hba1c": {
        "%": _IDENTITY,
        "mmol/mol": lambda value: HBA1C_NGSP_SLOPE * value + HBA1C_NGSP_INTERCEPT,
    },
    "creatinine": {
        "mg/dl": _IDENTITY,
        "umol/l": lambda value: value / UMOL_L_PER_MG_DL_CREATININE,
    },
    "egfr": {"ml/min/1.73m2": _IDENTITY},
    "sbp": {"mmhg": _IDENTITY},
    "resting_heart_rate": {"bpm": _IDENTITY, "count/min": _IDENTITY},
    "hrv_rmssd": {"ms": _IDENTITY},
    "vo2_max": {"ml/kg/min": _IDENTITY},
    "weight": {"kg": _IDENTITY, "lb": lambda value: value * KG_PER_LB},
}


@dataclass(frozen=True)
class Measurement:
    """One accepted value in canonical units.

    Attributes:
        record_id: The input record's id, or the main source record for a derived
            value.
        biomarker: Canonical biomarker or wearable metric name.
        value: Value in the canonical unit.
        measured_at: Measurement time, UTC.
        method: The record's measurement_method.
        derived_from: Source record ids when the engine derived this value.
    """

    record_id: str
    biomarker: str
    value: float
    measured_at: datetime
    method: str
    derived_from: tuple[str, ...] = ()

    @property
    def day(self) -> date:
        """Calendar day (UTC) of the measurement."""
        return self.measured_at.date()


@dataclass(frozen=True)
class Rejection:
    """A record that was not used, and why.

    Attributes:
        record_id: The record's id, if it had one.
        biomarker_abbr: The record's biomarker_abbr, if it had one.
        reason: Why the record was rejected.
    """

    record_id: str | None
    biomarker_abbr: str | None
    reason: str


@dataclass(frozen=True)
class IngestedRecords:
    """Accepted labs and wearable summaries, sorted by time, plus rejections.

    Attributes:
        labs: Measured lab and blood-pressure values, including derived ones.
        wearables: Wearable daily summaries.
        rejected: Records that were not used.
    """

    labs: list[Measurement] = field(default_factory=list)
    wearables: list[Measurement] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)


class RecordError(ValueError):
    """Raised for one record that cannot be accepted. The message is the reason."""


def _normalize_unit(unit: str) -> str:
    """Lower-case a unit, drop spaces, and spell micro and squared in ASCII."""
    cleaned = unit.strip().lower().replace(" ", "")
    for old, new in (("µ", "u"), ("μ", "u"), ("²", "2"), ("lbs", "lb")):
        cleaned = cleaned.replace(old, new)
    return cleaned


def parse_timestamp(raw: object) -> datetime:
    """Parse an ISO-8601 timestamp as UTC. A trailing "Z" is accepted.

    Args:
        raw: The measured_at field.

    Returns:
        A timezone-aware UTC datetime.

    Raises:
        RecordError: If raw is not an ISO-8601 string or datetime.
    """
    if isinstance(raw, datetime):
        stamp = raw
    elif isinstance(raw, str):
        try:
            stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as error:
            raise RecordError(f"measured_at is not ISO-8601: {raw!r}") from error
    else:
        raise RecordError("measured_at is missing or not a string")
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)


def convert_unit(biomarker: str, value: float, unit: str) -> float:
    """Convert a value to the biomarker's canonical unit.

    Example:
        >>> round(convert_unit("hdl", 1.3, "mmol/L"), 1)
        50.3

    Args:
        biomarker: Canonical biomarker name.
        value: Reported value.
        unit: Reported unit.

    Returns:
        The value in the canonical unit.

    Raises:
        RecordError: If the unit is not supported for this biomarker.
    """
    converter = UNIT_CONVERSIONS.get(biomarker, {}).get(_normalize_unit(unit))
    if converter is None:
        raise RecordError(f"unit {unit!r} is not supported for {biomarker}")
    return converter(value)


def parse_record(record: RawRecord) -> Measurement:
    """Validate and convert one record.

    Args:
        record: One record in the input shape.

    Returns:
        The accepted measurement.

    Raises:
        RecordError: With the rejection reason.
    """
    missing = [name for name in REQUIRED_FIELDS if record.get(name) in (None, "")]
    if missing:
        raise RecordError(f"missing fields: {', '.join(missing)}")
    abbr = str(record["biomarker_abbr"])
    biomarker_name = ALIAS_TO_BIOMARKER.get(normalize_name(abbr))
    if biomarker_name is None:
        raise RecordError(f"unknown biomarker_abbr {abbr!r}")
    biomarker = BIOMARKER_CATALOG[biomarker_name]
    if biomarker.kind == "unused":
        raise RecordError(f"{biomarker_name} is not used by the engine")
    raw_value = record["value"]
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        raise RecordError("value is not a number")
    if not math.isfinite(float(raw_value)):
        raise RecordError("value is not finite")
    method = str(record["measurement_method"])
    if biomarker.kind == "lab" and method not in MEASURED_LAB_METHODS:
        raise RecordError(f"measurement_method {method!r} is not a measured source")
    if biomarker.kind == "wearable" and not method.startswith(WEARABLE_METHOD_PREFIXES):
        raise RecordError(f"measurement_method {method!r} is not a wearable source")
    value = convert_unit(biomarker_name, float(raw_value), str(record["unit"]))
    if not is_plausible(biomarker_name, value):
        low, high = biomarker.plausible
        raise RecordError(
            f"{biomarker_name} {value:.4g} {biomarker.unit} is outside {low:g}-{high:g}"
        )
    return Measurement(
        record_id=str(record["id"]),
        biomarker=biomarker_name,
        value=value,
        measured_at=parse_timestamp(record["measured_at"]),
        method=method,
    )


def ingest_records(
    records: Sequence[RawRecord], age: float, sex: int
) -> IngestedRecords:
    """Accept, convert, and derive values from a batch of records.

    Args:
        records: Lab and wearable records in the input shape.
        age: The user's age in years, for eGFR.
        sex: NHANES sex code (1 = male, 2 = female), for eGFR.

    Returns:
        Accepted labs and wearables sorted by time, and rejections.
    """
    labs: list[Measurement] = []
    wearables: list[Measurement] = []
    rejected: list[Rejection] = []
    for record in records:
        try:
            measurement = parse_record(record)
        except RecordError as error:
            rejection = Rejection(
                record_id=_optional_str(record.get("id")),
                biomarker_abbr=_optional_str(record.get("biomarker_abbr")),
                reason=str(error),
            )
            logger.warning(
                "Rejected record %s (%s): %s",
                rejection.record_id,
                rejection.biomarker_abbr,
                rejection.reason,
            )
            rejected.append(rejection)
            continue
        kind = BIOMARKER_CATALOG[measurement.biomarker].kind
        (labs if kind == "lab" else wearables).append(measurement)
    labs += derive_values(labs, age, sex)
    return IngestedRecords(
        labs=sorted(labs, key=lambda item: item.measured_at),
        wearables=sorted(wearables, key=lambda item: item.measured_at),
        rejected=rejected,
    )


def derive_values(
    labs: Sequence[Measurement], age: float, sex: int
) -> list[Measurement]:
    """Non-HDL and eGFR the engine computes, never the extractor.

    Non-HDL is total cholesterol minus HDL from the same draw (same UTC day). eGFR
    uses CKD-EPI 2021 from creatinine. A derived value is skipped when the biomarker
    was reported directly that day, so one draw is counted once.

    Args:
        labs: Accepted lab measurements.
        age: Age in years.
        sex: NHANES sex code.

    Returns:
        The derived measurements.
    """
    reported = {(lab.biomarker, lab.day) for lab in labs}
    hdl_by_day = {lab.day: lab for lab in labs if lab.biomarker == "hdl"}
    derived: dict[tuple[str, date], Measurement] = {}
    for lab in labs:
        if lab.biomarker == "total_chol" and lab.day in hdl_by_day:
            hdl = hdl_by_day[lab.day]
            candidate = Measurement(
                record_id=lab.record_id,
                biomarker="non_hdl",
                value=lab.value - hdl.value,
                measured_at=lab.measured_at,
                method=lab.method,
                derived_from=(lab.record_id, hdl.record_id),
            )
        elif lab.biomarker == "creatinine" and sex in (1, 2):
            candidate = Measurement(
                record_id=lab.record_id,
                biomarker="egfr",
                value=egfr_ckd_epi_2021(lab.value, age, sex),
                measured_at=lab.measured_at,
                method=lab.method,
                derived_from=(lab.record_id,),
            )
        else:
            continue
        key = (candidate.biomarker, candidate.day)
        if key not in reported:
            derived.setdefault(key, candidate)
    return list(derived.values())


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
