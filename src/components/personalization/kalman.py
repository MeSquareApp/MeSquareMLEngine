"""One-dimensional Kalman filter for a single biomarker's latent level.

State x_t is the user's true level on the working scale (see BiomarkerSpec.transform).
    predict: x_t = x_{t-1} + u_t + w_t,   w_t ~ N(0, Q)
    update:  y   = x_t + e,               e   ~ N(0, R)   on each lab date
u_t is the daily step of the wearable effect. The effect moves toward
sum(beta * delta_feature) with time constant tau_days.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from scipy.signal import lfilter

from src.components.personalization.biomarker_config import BiomarkerSpec
from src.components.personalization.distributions import (
    FloatArray,
    daily_process_variance,
    forward,
    inverse,
    measurement_variance,
    natural_quantiles,
)

LabSeries = Sequence[tuple[date, float]]
HIGH_CONFIDENCE_RATIO = 0.35
MEDIUM_CONFIDENCE_RATIO = 0.7
MIN_LEVEL = 1e-6


@dataclass(frozen=True)
class Prior:
    """Population prior on the working scale.

    Attributes:
        mean: Prior mean.
        sd: Prior standard deviation.
    """

    mean: float
    sd: float


@dataclass(frozen=True)
class Innovation:
    """How a lab compared with the prediction made just before it.

    Attributes:
        date: Lab date.
        observed: Lab value in natural units.
        predicted: Predicted median in natural units just before the lab.
        z: Standardized residual. Roughly N(0, 1) when the filter is calibrated.
    """

    date: date
    observed: float
    predicted: float
    z: float


@dataclass(frozen=True)
class FilterResult:
    """Output of run_filter().

    Attributes:
        trajectory: One row per day: date, mean, sd, effect, basis, confidence,
            forecast, q10, q50, q90.
        innovations: One entry per lab.
        prior: The population prior the filter started from.
    """

    trajectory: pl.DataFrame
    innovations: list[Innovation]
    prior: Prior


def effect_path(spec: BiomarkerSpec, deltas: pl.DataFrame) -> FloatArray:
    """Lagged wearable effect in natural units, one value per day.

    The effect starts at zero and moves toward sum(beta * delta) by a fraction
    1 - exp(-1 / tau_days) each day.

    Args:
        spec: Biomarker settings with effect sizes and response time.
        deltas: Output of driver_deltas(), one row per day.

    Returns:
        Effect per day, same length as deltas.
    """
    target = np.zeros(deltas.height)
    for feature, beta in spec.effects.items():
        if feature in deltas.columns:
            target += beta * deltas[feature].to_numpy()
    if deltas.height < 2:
        return np.zeros_like(target)
    rate = 1.0 - np.exp(-1.0 / spec.tau_days)
    lagged = lfilter([rate], [1.0, rate - 1.0], target[1:])
    return np.concatenate(([0.0], lagged))


def confidence_label(sd: float, prior_sd: float) -> str:
    """Label the posterior's certainty relative to the population prior.

    Args:
        sd: Posterior standard deviation on the working scale.
        prior_sd: Prior standard deviation on the working scale.

    Returns:
        "high", "medium", or "low".
    """
    ratio = sd / prior_sd if prior_sd > 0 else 1.0
    if ratio < HIGH_CONFIDENCE_RATIO:
        return "high"
    if ratio < MEDIUM_CONFIDENCE_RATIO:
        return "medium"
    return "low"


def run_filter(
    spec: BiomarkerSpec,
    prior: Prior,
    labs: LabSeries,
    deltas: pl.DataFrame,
    as_of: date,
) -> FilterResult:
    """Filter one biomarker over every day in deltas. Days after as_of are forecast.

    Args:
        spec: Biomarker settings.
        prior: Population prior on the working scale.
        labs: Lab dates and values in natural units.
        deltas: Output of driver_deltas(). Its dates define the timeline.
        as_of: Last observed day.

    Returns:
        Daily trajectory, lab innovations, and the prior.
    """
    labs_by_day: defaultdict[date, list[float]] = defaultdict(list)
    for lab_day, value in labs:
        labs_by_day[lab_day].append(value)

    days: list[date] = deltas["date"].to_list()
    effect = effect_path(spec, deltas)
    steps = np.diff(effect, prepend=0.0)
    means = np.empty(len(days))
    sds = np.empty(len(days))
    bases: list[str] = []
    innovations: list[Innovation] = []
    mean, var = prior.mean, prior.sd**2
    seen_lab = False

    for i, day in enumerate(days):
        if i > 0:
            level = float(inverse(mean, spec.transform))
            mean += _working_step(level, float(steps[i]), spec)
            var += daily_process_variance(level, spec.annual_drift_cv, spec.transform)
        for value in labs_by_day.get(day, []):
            mean, var, innovation = _update(spec, mean, var, (day, value))
            innovations.append(innovation)
        seen_lab = seen_lab or day in labs_by_day
        bases.append(_basis(day in labs_by_day, seen_lab, float(effect[i])))
        means[i], sds[i] = mean, np.sqrt(var)

    quantiles = natural_quantiles(means, sds, spec.transform)
    low, high = spec.clamp
    trajectory = pl.DataFrame(
        {
            "date": days,
            "mean": means,
            "sd": sds,
            "effect": effect,
            "basis": bases,
            "confidence": [confidence_label(sd, prior.sd) for sd in sds],
            "forecast": [day > as_of for day in days],
            **{name: np.clip(values, low, high) for name, values in quantiles.items()},
        }
    )
    return FilterResult(trajectory, innovations, prior)


def _working_step(level: float, step: float, spec: BiomarkerSpec) -> float:
    """Convert a natural-unit effect step into a working-scale step."""
    if spec.transform == "log":
        return float(np.log(max(level + step, MIN_LEVEL) / level))
    return step


def _update(
    spec: BiomarkerSpec, mean: float, var: float, lab: tuple[date, float]
) -> tuple[float, float, Innovation]:
    """Apply one lab measurement. Returns the new mean, variance, and innovation."""
    lab_day, value = lab
    observed = float(forward(value, spec.transform))
    noise = measurement_variance(value, spec.measurement_cv, spec.transform)
    innovation = Innovation(
        date=lab_day,
        observed=value,
        predicted=float(inverse(mean, spec.transform)),
        z=(observed - mean) / float(np.sqrt(var + noise)),
    )
    gain = var / (var + noise)
    return mean + gain * (observed - mean), var * (1.0 - gain), innovation


def _basis(measured: bool, seen_lab: bool, effect: float) -> str:
    """Where a day's estimate comes from."""
    if measured:
        return "measured"
    if seen_lab:
        return "anchored"
    return "estimated" if effect != 0.0 else "prior"
