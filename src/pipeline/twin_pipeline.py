"""Digital twin: population prior, lab updates, wearable drift, then PREVENT risk.

Stateless: the caller sends the user's full lab and wearable history each time, and the
twin is recomputed from scratch. Each biomarker runs its own 1-D filter, so this is
cheap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Protocol

import numpy as np
import polars as pl

from src.components.egfr import egfr_ckd_epi_2021
from src.components.personalization.biomarker_config import BIOMARKERS
from src.components.personalization.distributions import (
    FloatArray,
    inverse,
    normal_from_quantiles,
)
from src.components.personalization.kalman import FilterResult, Prior, run_filter
from src.components.personalization.wearable_features import (
    WearableRecord,
    WearableSummary,
    driver_deltas,
    summarize,
)
from src.components.prevent import (
    AGE_RANGE,
    PreventInputs,
    clip_to_validated,
    prevent_available,
    prevent_risk,
    summarize_risk,
)

ProfileValue = str | float | int | None
Profile = Mapping[str, ProfileValue]
LabRecord = Mapping[str, date | str | float]
RiskSummary = dict[str, dict[str, float]]

DEFAULT_HORIZON_DAYS = 90
# Each sample is one row through the official R package, which evaluates rows one at
# a time.
DEFAULT_SAMPLES = 500
HBA1C_DIABETES = 6.5
PREVENT_BIOMARKERS = ("non_hdl", "hdl", "sbp", "egfr", "hba1c")
CM_PER_M = 100.0


class QuantilePrior(Protocol):
    """Anything that returns population quantiles for one profile."""

    def predict_quantiles(self, record: Profile) -> dict[str, dict[float, float]]:
        """Population quantiles keyed by target, then tau."""
        ...


@dataclass(frozen=True)
class TwinInputs:
    """Everything the backend sends for one user.

    Attributes:
        profile: Feature record with NHANES codes, plus optional height_cm.
        labs: Lab rows with "date", "biomarker", and "value".
        wearable_days: End-of-day wearable records.
        as_of: Last observed day. Defaults to the latest data date.
        horizon_days: Days to forecast past as_of.
    """

    profile: Profile
    labs: Sequence[LabRecord] = field(default_factory=tuple)
    wearable_days: Sequence[WearableRecord] = field(default_factory=tuple)
    as_of: date | None = None
    horizon_days: int = DEFAULT_HORIZON_DAYS


@dataclass
class TwinResult:
    """Personalized biomarker trajectories and PREVENT risk for one user.

    Attributes:
        as_of: Last observed day.
        prior_quantiles: Population quantiles keyed by biomarker, then tau.
        biomarkers: Filter output per biomarker.
        wearable: Wearable baseline and trailing-window summary.
        prevent: Risk summaries keyed "population_prior", "current", "forecast".
        prevent_note: Why PREVENT is empty, if it is.
    """

    as_of: date
    prior_quantiles: dict[str, dict[float, float]]
    biomarkers: dict[str, FilterResult]
    wearable: WearableSummary
    prevent: dict[str, RiskSummary] = field(default_factory=dict)
    prevent_note: str | None = None


@dataclass(frozen=True)
class _PreventContext:
    """Per-user PREVENT inputs that are fixed across Monte-Carlo samples."""

    age: float
    sex: int
    height_m: float
    prior_weight: float
    latest_weight: float
    diagnosed_diabetes: bool
    smoking: float
    bp_treated: float
    statin: float


def normalize_labs(
    labs: Sequence[LabRecord], profile: Profile
) -> dict[str, list[tuple[date, float]]]:
    """Map lab rows onto filter biomarkers, one measurement per blood draw.

    total_chol becomes non_hdl when HDL is on the same date. scr becomes eGFR
    (race-free CKD-EPI 2021). A derived value is used only when the same biomarker
    was not reported directly that day, and at most once per day, so one draw is
    never counted twice. Other biomarkers are ignored.

    Args:
        labs: Lab rows with "date", "biomarker", and "value".
        profile: Needs "age" and "sex" to convert creatinine.

    Returns:
        (date, value) pairs per biomarker in BIOMARKERS.
    """
    by_name: dict[str, list[tuple[date, float]]] = {name: [] for name in BIOMARKERS}
    rows = [
        (_as_date(row["date"]), str(row["biomarker"]), _as_float(row["value"]))
        for row in labs
    ]
    hdl_by_date = {day: value for day, name, value in rows if name == "hdl"}
    derived: dict[tuple[str, date], float] = {}
    for day, name, value in rows:
        if name in by_name:
            by_name[name].append((day, value))
        elif name == "total_chol" and day in hdl_by_date:
            derived.setdefault(("non_hdl", day), value - hdl_by_date[day])
        elif name == "scr":
            derived.setdefault(
                ("egfr", day),
                egfr_ckd_epi_2021(
                    value, _as_float(profile["age"]), int(_as_float(profile["sex"]))
                ),
            )
    reported = {(name, day) for day, name, _ in rows}
    for (name, day), value in derived.items():
        if (name, day) not in reported:
            by_name[name].append((day, value))
    return by_name


def predict_twin(
    prior: QuantilePrior,
    inputs: TwinInputs,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
) -> TwinResult:
    """Build the digital twin for one user.

    Args:
        prior: Population quantile model.
        inputs: Profile, labs, wearables, as_of, and horizon.
        n_samples: Monte-Carlo samples per PREVENT view.
        seed: Random seed for the Monte-Carlo draws.

    Returns:
        Trajectories per biomarker, wearable summary, and PREVENT views.
    """
    as_of, start = _timeline(inputs)
    end = as_of + timedelta(days=inputs.horizon_days)
    wearable = summarize(inputs.wearable_days, as_of)

    prior_profile = dict(inputs.profile)
    if prior_profile.get("weight_kg") is None:
        prior_profile["weight_kg"] = wearable.baseline.get("weight_kg")
    prior_quantiles = prior.predict_quantiles(prior_profile)

    deltas = driver_deltas(wearable, start, end)
    lab_values = normalize_labs(inputs.labs, inputs.profile)
    results = {
        name: run_filter(
            spec,
            Prior(*normal_from_quantiles(prior_quantiles[name], spec.transform)),
            lab_values[name],
            deltas,
            as_of,
        )
        for name, spec in BIOMARKERS.items()
        if name in prior_quantiles
    }
    twin = TwinResult(as_of, prior_quantiles, results, wearable)
    context, twin.prevent_note = _prevent_context(twin, prior_profile)
    if context is not None:
        twin.prevent = {
            view: _risk_on(twin, context, (on, weight), (n_samples, seed))
            for view, on, weight in (
                ("population_prior", None, context.prior_weight),
                ("current", as_of, context.latest_weight),
                ("forecast", end, context.latest_weight),
            )
        }
    return twin


def biomarker_samples(
    twin: TwinResult, on: date | None, n_samples: int, seed: int
) -> dict[str, FloatArray]:
    """Draw natural-unit samples per biomarker.

    Args:
        twin: Output of predict_twin().
        on: Day to sample from the posterior. None samples the population prior.
        n_samples: Number of draws per biomarker.
        seed: Random seed.

    Returns:
        Samples per biomarker, clipped to each biomarker's plausible range.

    Raises:
        ValueError: If on is outside the twin's timeline.
    """
    rng = np.random.default_rng(seed)
    samples: dict[str, FloatArray] = {}
    for name, result in twin.biomarkers.items():
        spec = BIOMARKERS[name]
        if on is None:
            mean, sd = result.prior.mean, result.prior.sd
        else:
            rows = result.trajectory.filter(pl.col("date") == on)
            if rows.is_empty():
                raise ValueError(f"{on} is outside the twin timeline")
            mean, sd = rows["mean"].item(), rows["sd"].item()
        draws = inverse(rng.normal(mean, sd, n_samples), spec.transform)
        samples[name] = np.clip(draws, *spec.clamp)
    return samples


def _timeline(inputs: TwinInputs) -> tuple[date, date]:
    """Return (as_of, start). Start is the earliest data date, at most as_of."""
    data_dates = [_as_date(row["date"]) for row in inputs.labs]
    data_dates += [_as_date(row["date"]) for row in inputs.wearable_days]
    as_of = inputs.as_of or (max(data_dates) if data_dates else date.today())
    return as_of, min([*data_dates, as_of])


def _prevent_context(
    twin: TwinResult, profile: Profile
) -> tuple[_PreventContext | None, str | None]:
    """Fixed PREVENT inputs, or a note explaining why PREVENT cannot run."""
    if not prevent_available():
        return None, "AHA PREVENT R package is not available"
    age = _as_float(profile["age"])
    if not AGE_RANGE[0] <= age <= AGE_RANGE[1]:
        return None, f"PREVENT is valid for ages {AGE_RANGE[0]}-{AGE_RANGE[1]}"
    missing = [name for name in PREVENT_BIOMARKERS if name not in twin.biomarkers]
    if missing:
        return None, f"Missing biomarker models: {', '.join(missing)}"
    height = profile.get("height_cm")
    if height is None:
        return None, "height_cm is required for BMI"
    prior_weight = profile.get("weight_kg")
    latest_weight = _latest_weight(twin.wearable) or prior_weight
    if prior_weight is None or latest_weight is None:
        return None, "weight_kg is required for BMI"
    context = _PreventContext(
        age=age,
        sex=int(_as_float(profile["sex"])),
        height_m=_as_float(height) / CM_PER_M,
        prior_weight=_as_float(prior_weight),
        latest_weight=_as_float(latest_weight),
        diagnosed_diabetes=profile.get("diabetes_dx") == "yes"
        or profile.get("insulin") == 1
        or profile.get("diabetes_pills") == 1,
        smoking=1.0 if profile.get("smoking") == "current" else 0.0,
        bp_treated=float(profile.get("bp_meds") or 0),
        statin=float(profile.get("chol_meds") or 0),
    )
    return context, None


def _latest_weight(wearable: WearableSummary) -> float | None:
    """Latest 30-day mean weight from wearables, if any."""
    if "weight_kg_30d" not in wearable.rolling.columns:
        return None
    latest = wearable.rolling["weight_kg_30d"].drop_nulls()
    return float(latest[-1]) if latest.len() else None


def _risk_on(
    twin: TwinResult,
    context: _PreventContext,
    view: tuple[date | None, float],
    sampling: tuple[int, int],
) -> RiskSummary:
    """Monte-Carlo PREVENT risk for one (date, weight) view."""
    on, weight = view
    samples = biomarker_samples(twin, on, *sampling)
    diabetes = context.diagnosed_diabetes | (samples["hba1c"] >= HBA1C_DIABETES)
    total_chol = samples["non_hdl"] + samples["hdl"]
    risk = prevent_risk(
        PreventInputs(
            age=context.age,
            sex=context.sex,
            total_chol=clip_to_validated("total_chol", total_chol),
            hdl=clip_to_validated("hdl", samples["hdl"]),
            sbp=clip_to_validated("sbp", samples["sbp"]),
            bmi=clip_to_validated("bmi", weight / context.height_m**2),
            egfr=samples["egfr"],
            diabetes=diabetes.astype(np.float64),
            smoking=context.smoking,
            bp_treated=context.bp_treated,
            statin=context.statin,
        )
    )
    return summarize_risk(risk)


def _as_date(value: object) -> date:
    """Accept a date or an ISO date string.

    Raises:
        ValueError: If value is neither.
    """
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"Not a date: {value!r}")


def _as_float(value: object) -> float:
    """Accept an int, float, or numeric string.

    Raises:
        ValueError: If value is not numeric.
    """
    if isinstance(value, int | float | str) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"Not a number: {value!r}")
