"""Digital twin: population prior → lab updates → wearable drift → PREVENT risk.

Stateless: the caller sends the user's full lab and wearable history each time, and the twin is
recomputed from scratch. Each biomarker runs its own 1-D filter, so this is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from src.components.egfr import egfr_ckd_epi_2009
from src.components.personalization.biomarker_config import BIOMARKERS
from src.components.personalization.distributions import inverse, normal_from_quantiles
from src.components.personalization.kalman import FilterResult, run_filter
from src.components.personalization.wearable_features import (
    WearableSummary,
    driver_deltas,
    summarize,
)
from src.components.prevent import (
    AGE_RANGE,
    PreventInputs,
    clip_to_validated,
    prevent_risk,
    summarize_risk,
)

DEFAULT_HORIZON_DAYS = 90
DEFAULT_SAMPLES = 2000
HBA1C_DIABETES = 6.5


class QuantilePrior(Protocol):
    def predict_quantiles(self, profile: pd.DataFrame) -> dict[str, dict[float, float]]: ...


@dataclass
class TwinResult:
    as_of: date
    prior_quantiles: dict[str, dict[float, float]]
    biomarkers: dict[str, FilterResult]
    wearable: WearableSummary
    prevent: dict[str, dict[str, dict[str, float]] | None] = field(default_factory=dict)
    prevent_note: str | None = None


def normalize_labs(labs: list[dict], profile: dict) -> dict[str, list[tuple[date, float]]]:
    """Map lab rows onto filter biomarkers.

    total_chol becomes non_hdl when HDL is on the same date. scr becomes eGFR (CKD-EPI 2009).
    """
    by_name: dict[str, list[tuple[date, float]]] = {name: [] for name in BIOMARKERS}
    hdl_by_date = {
        _as_date(row["date"]): float(row["value"]) for row in labs if row["biomarker"] == "hdl"
    }
    for row in labs:
        day, value, name = _as_date(row["date"]), float(row["value"]), row["biomarker"]
        if name in by_name:
            by_name[name].append((day, value))
        elif name == "total_chol" and day in hdl_by_date:
            by_name["non_hdl"].append((day, value - hdl_by_date[day]))
        elif name == "scr":
            by_name["egfr"].append(
                (
                    day,
                    egfr_ckd_epi_2009(
                        value, float(profile["age"]), int(profile["sex"]), int(profile["race_eth"])
                    ),
                )
            )
    return by_name


def predict_twin(
    prior: QuantilePrior,
    profile: dict,
    labs: list[dict] | None = None,
    wearable_days: pd.DataFrame | None = None,
    as_of: date | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
) -> TwinResult:
    labs = labs or []
    wearable_days = wearable_days if wearable_days is not None else pd.DataFrame()

    data_dates = [_as_date(row["date"]) for row in labs]
    if not wearable_days.empty:
        data_dates += [d.date() for d in pd.to_datetime(wearable_days["date"])]
    as_of = as_of or (max(data_dates) if data_dates else date.today())
    start = min([*data_dates, as_of])
    end = as_of + timedelta(days=horizon_days)

    wearable = summarize(wearable_days, as_of)
    prior_profile = dict(profile)
    if prior_profile.get("weight_kg") is None and wearable.baseline.get("weight_kg") is not None:
        prior_profile["weight_kg"] = wearable.baseline["weight_kg"]
    prior_quantiles = prior.predict_quantiles(pd.DataFrame([prior_profile]))

    deltas = driver_deltas(wearable, start, end)
    lab_values = normalize_labs(labs, profile)
    results: dict[str, FilterResult] = {}
    for name, spec in BIOMARKERS.items():
        if name not in prior_quantiles:
            continue
        mean, sd = normal_from_quantiles(prior_quantiles[name], spec.transform)
        results[name] = run_filter(spec, mean, sd, lab_values[name], deltas, as_of)

    twin = TwinResult(as_of, prior_quantiles, results, wearable)
    twin.prevent, twin.prevent_note = _prevent_views(twin, prior_profile, end, n_samples, seed)
    return twin


def biomarker_samples(
    twin: TwinResult, on: date | None, n_samples: int, seed: int
) -> dict[str, np.ndarray]:
    """Natural-unit samples per biomarker. on=None samples the population prior."""
    rng = np.random.default_rng(seed)
    samples = {}
    for name, result in twin.biomarkers.items():
        spec = BIOMARKERS[name]
        if on is None:
            mean, sd = result.prior_mean, result.prior_sd
        else:
            row = result.trajectory.loc[result.trajectory["date"] == on].iloc[0]
            mean, sd = float(row["mean"]), float(row["sd"])
        draws = inverse(rng.normal(mean, sd, n_samples), spec.transform)
        samples[name] = np.clip(draws, *spec.clamp)
    return samples


def _prevent_views(
    twin: TwinResult, profile: dict, end: date, n_samples: int, seed: int
) -> tuple[dict, str | None]:
    age = float(profile["age"])
    if not AGE_RANGE[0] <= age <= AGE_RANGE[1]:
        return {}, f"PREVENT is valid for ages {AGE_RANGE[0]}-{AGE_RANGE[1]}"
    missing = [name for name in ("non_hdl", "hdl", "sbp", "egfr", "hba1c") if name not in twin.biomarkers]
    if missing:
        return {}, f"Missing biomarker models: {', '.join(missing)}"
    height = profile.get("height_cm")
    if not height:
        return {}, "height_cm is required for BMI"

    latest_weight = profile.get("weight_kg")
    recent = twin.wearable.rolling.get("weight_kg_30d") if not twin.wearable.rolling.empty else None
    if recent is not None and recent.notna().any():
        latest_weight = float(recent.dropna().iloc[-1])
    if latest_weight is None:
        return {}, "weight_kg is required for BMI"

    diagnosed = (
        profile.get("diabetes_dx") == "yes"
        or profile.get("insulin") == 1
        or profile.get("diabetes_pills") == 1
    )
    views = {}
    for view, on, weight in (
        ("population_prior", None, profile.get("weight_kg") or latest_weight),
        ("current", twin.as_of, latest_weight),
        ("forecast", end, latest_weight),
    ):
        s = biomarker_samples(twin, on, n_samples, seed)
        risk = prevent_risk(
            PreventInputs(
                age=age,
                sex=int(profile["sex"]),
                total_chol=clip_to_validated("total_chol", s["non_hdl"] + s["hdl"]),
                hdl=clip_to_validated("hdl", s["hdl"]),
                sbp=clip_to_validated("sbp", s["sbp"]),
                bmi=clip_to_validated("bmi", float(weight) / (float(height) / 100.0) ** 2),
                egfr=s["egfr"],
                diabetes=np.where(diagnosed | (s["hba1c"] >= HBA1C_DIABETES), 1.0, 0.0),
                smoking=1.0 if profile.get("smoking") == "current" else 0.0,
                bp_treated=float(profile.get("bp_meds") or 0),
                statin=float(profile.get("chol_meds") or 0),
            )
        )
        views[view] = summarize_risk(risk)
    return views, None


def _as_date(value) -> date:
    return value if isinstance(value, date) else pd.Timestamp(value).date()
