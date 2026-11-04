"""Canonical biomarkers and wearable metrics: names, units, aliases, plausible ranges.

One table serves lab ingestion, wearable ingestion and the training outlier rule, so
a value the API would reject is never used to train the prior either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MG_DL_PER_MMOL_L_CHOLESTEROL = 38.67
UMOL_L_PER_MG_DL_CREATININE = 88.42
KG_PER_LB = 0.45359237
# NGSP (%) from IFCC (mmol/mol): Hoelzel et al., Clin Chem 2004.
HBA1C_NGSP_SLOPE = 0.09148
HBA1C_NGSP_INTERCEPT = 2.152

RecordKind = Literal["lab", "wearable", "unused"]


@dataclass(frozen=True)
class Biomarker:
    """One measurable quantity the engine accepts.

    Attributes:
        name: Canonical name used everywhere in the engine.
        unit: Canonical unit every value is converted to.
        kind: "lab" for blood tests and BP, "wearable" for daily device summaries,
            "unused" for known names the engine does not model.
        plausible: Inclusive (low, high) range in the canonical unit.
        aliases: Accepted biomarker_abbr spellings, normalized (see normalize_name).
    """

    name: str
    unit: str
    kind: RecordKind
    plausible: tuple[float, float] = (float("-inf"), float("inf"))
    aliases: tuple[str, ...] = field(default_factory=tuple)


BIOMARKER_CATALOG: dict[str, Biomarker] = {
    biomarker.name: biomarker
    for biomarker in (
        Biomarker(
            "total_chol",
            "mg/dL",
            "lab",
            (50.0, 500.0),
            ("tc", "chol", "cholesterol", "total_cholesterol", "cholesterol_total"),
        ),
        Biomarker("hdl", "mg/dL", "lab", (5.0, 200.0), ("hdl_c", "hdl_cholesterol")),
        Biomarker("non_hdl", "mg/dL", "lab", (20.0, 450.0), ("non_hdl_c", "nonhdl")),
        Biomarker(
            "hba1c",
            "%",
            "lab",
            (3.0, 20.0),
            ("a1c", "hemoglobin_a1c", "glycated_hemoglobin", "glycohemoglobin"),
        ),
        Biomarker(
            "creatinine",
            "mg/dL",
            "lab",
            (0.1, 20.0),
            ("creat", "scr", "serum_creatinine", "creatinine_serum"),
        ),
        Biomarker("egfr", "mL/min/1.73m2", "lab", (2.0, 200.0), ("gfr", "e_gfr")),
        Biomarker(
            "sbp",
            "mmHg",
            "lab",
            (60.0, 260.0),
            ("systolic", "systolic_bp", "systolic_blood_pressure"),
        ),
        Biomarker(
            "resting_heart_rate",
            "bpm",
            "wearable",
            (25.0, 150.0),
            ("rhr", "resting_hr", "restingheartrate"),
        ),
        Biomarker("hrv_rmssd", "ms", "wearable", (3.0, 300.0), ("hrv", "rmssd")),
        Biomarker("vo2_max", "mL/kg/min", "wearable", (8.0, 95.0), ("vo2max",)),
        Biomarker(
            "weight", "kg", "wearable", (25.0, 350.0), ("body_weight", "weight_kg")
        ),
        Biomarker("ldl", "mg/dL", "unused", aliases=("ldl_c", "ldl_cholesterol")),
        Biomarker("triglycerides", "mg/dL", "unused", aliases=("tg", "trig")),
    )
}

# Prior-model targets and the biomarker each one is measured as.
MODEL_BIOMARKERS = ("non_hdl", "hdl", "hba1c", "egfr", "sbp")

# Measurement methods whose values count as measured for PREVENT.
MEASURED_LAB_METHODS = frozenset(
    {
        "blood_test_lab_report_upload",
        "lab_report_upload",
        "lab_integration",
        "clinic_measurement",
        "home_bp_protocol",
    }
)
WEARABLE_METHOD_PREFIXES = ("wearable_", "smart_scale")


def normalize_name(raw: str) -> str:
    """Lower-case a biomarker name and turn every non-alphanumeric run into "_".

    Example:
        >>> normalize_name(" HDL-C ")
        'hdl_c'

    Args:
        raw: Name as the extractor or device wrote it.

    Returns:
        The normalized name.
    """
    cleaned = "".join(char if char.isalnum() else "_" for char in raw.strip().lower())
    return "_".join(part for part in cleaned.split("_") if part)


ALIAS_TO_BIOMARKER: dict[str, str] = {
    alias: biomarker.name
    for biomarker in BIOMARKER_CATALOG.values()
    for alias in (normalize_name(biomarker.name), *biomarker.aliases)
}


def is_plausible(biomarker: str, value: float) -> bool:
    """Whether a canonical-unit value is inside the biomarker's plausible range.

    Args:
        biomarker: Canonical biomarker name.
        value: Value in the canonical unit.

    Returns:
        True when low <= value <= high.

    Raises:
        KeyError: If the biomarker is unknown.
    """
    low, high = BIOMARKER_CATALOG[biomarker].plausible
    return low <= value <= high
