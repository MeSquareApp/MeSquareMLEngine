"""Per-biomarker settings for the personalization filter. Tune here, nowhere else.

Effects are steady-state changes in the biomarker's own unit per one unit of wearable change
relative to the user's own baseline. They come from intervention meta-analyses and are
deliberately conservative. They are priors to be replaced by learned per-user effects once
paired data exists.

- weight_kg: Dattilo & Kris-Etherton 1992 (lipids per kg), Neter 2003 (about 1 mmHg per kg).
- steps (per 1000 steps/day): Kodama 2007 (HDL), Cornelissen 2013 (aerobic SBP), scaled from
  structured-training doses to daily steps.
- resting_hr (per bpm): fitness marker. Small positive association with SBP.
- sleep_h (per hour): short-sleep associations with SBP and glycaemia. Kept small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Transform = Literal["identity", "log"]


@dataclass(frozen=True)
class BiomarkerSpec:
    name: str
    unit: str
    transform: Transform
    # Analytical plus within-person biological CV of a single measurement.
    measurement_cv: float
    # Long-run within-person drift over one year without any behaviour change, as a CV.
    annual_drift_cv: float
    # First-order response time of the biomarker to a sustained behaviour change.
    tau_days: float
    effects: dict[str, float] = field(default_factory=dict)
    clamp: tuple[float, float] = (0.0, float("inf"))


BIOMARKERS: dict[str, BiomarkerSpec] = {
    "non_hdl": BiomarkerSpec(
        name="non_hdl",
        unit="mg/dL",
        transform="identity",
        measurement_cv=0.07,
        annual_drift_cv=0.08,
        tau_days=30.0,
        effects={"weight_kg": 1.5, "steps": -0.5, "resting_hr": 0.1},
        clamp=(20.0, 400.0),
    ),
    "hdl": BiomarkerSpec(
        name="hdl",
        unit="mg/dL",
        transform="identity",
        measurement_cv=0.07,
        annual_drift_cv=0.06,
        tau_days=45.0,
        effects={"weight_kg": -0.35, "steps": 0.3, "resting_hr": -0.05},
        clamp=(10.0, 150.0),
    ),
    "hba1c": BiomarkerSpec(
        name="hba1c",
        unit="%",
        transform="log",
        measurement_cv=0.025,
        annual_drift_cv=0.04,
        tau_days=60.0,
        effects={"weight_kg": 0.02, "steps": -0.01, "sleep_h": -0.02},
        clamp=(3.5, 18.0),
    ),
    "egfr": BiomarkerSpec(
        name="egfr",
        unit="mL/min/1.73m2",
        transform="log",
        measurement_cv=0.05,
        annual_drift_cv=0.04,
        tau_days=180.0,
        effects={},
        clamp=(5.0, 160.0),
    ),
    "sbp": BiomarkerSpec(
        name="sbp",
        unit="mmHg",
        transform="identity",
        measurement_cv=0.06,
        annual_drift_cv=0.05,
        tau_days=21.0,
        effects={"weight_kg": 1.0, "steps": -0.4, "resting_hr": 0.2, "sleep_h": -0.5},
        clamp=(80.0, 220.0),
    ),
}

WEARABLE_FEATURES = ("steps", "resting_hr", "sleep_h", "hrv_rmssd", "mvpa_min", "weight_kg")
# Features that drive biomarkers. The others are summarized for display only.
DRIVER_FEATURES = ("weight_kg", "steps", "resting_hr", "sleep_h")
# Steps are modelled per 1000 steps/day.
FEATURE_SCALE = {"steps": 1000.0}

BASELINE_DAYS = 30
MIN_VALID_DAYS = 20
