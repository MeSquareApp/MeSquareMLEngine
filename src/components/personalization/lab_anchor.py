"""Lab-anchored bands, real-change detection, and anchor resets.

Stage 1 (first lab): the band is centred on the latest measured lab. Its width is
what separates two measurements of the same person: sqrt(2) x the measurement SD
(analytical plus within-person variation), plus drift that grows with the lab's age.
So the 80% band is a prediction of the user's next lab.

Stage 2 (repeat labs): the latest lab still anchors. The previous lab is compared
with it using the 95% reference change value (RCV); a change beyond it is flagged as
real, which keeps false alerts near 5% when nothing changed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from src.components.personalization.biomarker_config import (
    BIOMARKERS,
    BP_ANALYTES,
    LIPID_ANALYTES,
)
from src.components.personalization.distributions import (
    Z_95,
    band,
    drift_sigma,
    log_sigma,
)
from src.components.record_ingestion import Measurement


@dataclass(frozen=True)
class TreatmentChanges:
    """Profile facts that make an older lab stop describing the user.

    Attributes:
        statin_started_at: Day a statin was started, if known.
        bp_meds_started_at: Day BP medication was started, if known.
        pregnant: Whether the user is pregnant now.
    """

    statin_started_at: date | None = None
    bp_meds_started_at: date | None = None
    pregnant: bool = False


@dataclass(frozen=True)
class ChangeFlag:
    """Latest lab compared with the previous one, using the 95% RCV.

    Attributes:
        previous_lab_id: Record id of the previous lab.
        previous_value: Previous value in the canonical unit.
        previous_measured_at: Day of the previous lab.
        percent_change: Latest vs previous, in percent.
        rcv_low_pct: Largest fall consistent with noise alone, in percent.
        rcv_high_pct: Largest rise consistent with noise alone, in percent.
        significant: Whether the change is beyond the RCV, so likely real.
    """

    previous_lab_id: str
    previous_value: float
    previous_measured_at: date
    percent_change: float
    rcv_low_pct: float
    rcv_high_pct: float
    significant: bool


def lab_sigma(analyte: str, days_since_lab: float) -> float:
    """Log-scale SD of the user's next lab around their latest lab.

    Args:
        analyte: Canonical analyte name.
        days_since_lab: Age of the latest lab in days.

    Returns:
        sqrt(2 * measurement SD^2 + drift SD^2).
    """
    spec = BIOMARKERS[analyte]
    measurement = log_sigma(spec.measurement_cv)
    drift = drift_sigma(spec.annual_drift_cv, days_since_lab)
    return math.sqrt(2.0 * measurement**2 + drift**2)


def lab_band(lab: Measurement, as_of: date) -> tuple[float, float, float]:
    """p10, p50, p90 of the user's next lab, anchored on their latest lab.

    Example:
        A lab of 50 mg/dL HDL taken today gives roughly (44.3, 50.0, 56.4).

    Args:
        lab: The anchoring measurement.
        as_of: Day the band is for. The band widens as the lab ages.

    Returns:
        (p10, p50, p90) in the canonical unit.
    """
    days = (as_of - lab.day).days
    return band(lab.value, lab_sigma(lab.analyte, days))


def reference_change(analyte: str) -> tuple[float, float]:
    """Two-sided 95% reference change value of the analyte, as fractions.

    Uses the log-normal form (Fokkema et al., Clin Chem 2006), so a fall and a rise
    have different sizes.

    Args:
        analyte: Canonical analyte name.

    Returns:
        (largest fall, largest rise) explained by measurement noise alone.
    """
    limit = Z_95 * math.sqrt(2.0) * log_sigma(BIOMARKERS[analyte].measurement_cv)
    return math.exp(-limit) - 1.0, math.exp(limit) - 1.0


def change_flag(previous: Measurement, latest: Measurement) -> ChangeFlag:
    """Compare two labs of one analyte against the 95% RCV.

    Args:
        previous: The earlier lab.
        latest: The later lab.

    Returns:
        The change and whether it exceeds the RCV.
    """
    low, high = reference_change(latest.analyte)
    change = latest.value / previous.value - 1.0
    return ChangeFlag(
        previous_lab_id=previous.record_id,
        previous_value=previous.value,
        previous_measured_at=previous.day,
        percent_change=100.0 * change,
        rcv_low_pct=100.0 * low,
        rcv_high_pct=100.0 * high,
        significant=not low <= change <= high,
    )


def anchoring_labs(
    analyte: str,
    labs: Sequence[Measurement],
    as_of: date,
    treatments: TreatmentChanges,
) -> list[Measurement]:
    """Labs of an analyte that still describe the user, oldest first.

    A lab taken after as_of is not observed yet. Pregnancy drops every lab. Starting
    a statin drops lipid labs taken before it; starting BP medication drops blood
    pressure readings taken before it. With nothing left, the analyte returns to
    the population prior until a new lab arrives.

    Args:
        analyte: Canonical analyte name.
        labs: Accepted lab measurements.
        as_of: Day of the prediction.
        treatments: Treatment changes from the profile.

    Returns:
        The usable labs of this analyte, sorted by time.
    """
    if treatments.pregnant:
        return []
    started = None
    if analyte in LIPID_ANALYTES:
        started = treatments.statin_started_at
    elif analyte in BP_ANALYTES:
        started = treatments.bp_meds_started_at
    return sorted(
        (
            lab
            for lab in labs
            if lab.analyte == analyte
            and lab.day <= as_of
            and (started is None or lab.day >= started)
        ),
        key=lambda lab: lab.measured_at,
    )
