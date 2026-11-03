"""One-dimensional Kalman filter for a single biomarker's latent level.

State x_t is the user's true level on the working scale (see BiomarkerSpec.transform).
    predict: x_t = x_{t-1} + u_t + w_t,   w_t ~ N(0, Q)
    update:  y   = x_t + e,               e   ~ N(0, R)   on each lab date
u_t is the daily step of the wearable effect. The effect moves toward
sum(beta * delta_feature) with time constant tau_days.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.components.personalization.biomarker_config import BiomarkerSpec
from src.components.personalization.distributions import (
    daily_process_variance,
    forward,
    inverse,
    measurement_variance,
    natural_quantiles,
)


@dataclass(frozen=True)
class Innovation:
    date: date
    observed: float
    predicted: float
    z: float


@dataclass(frozen=True)
class FilterResult:
    trajectory: pd.DataFrame
    innovations: list[Innovation]
    prior_mean: float
    prior_sd: float


def effect_path(spec: BiomarkerSpec, deltas: pd.DataFrame) -> pd.Series:
    """Lagged wearable effect in natural units, one value per day."""
    target = pd.Series(0.0, index=deltas.index)
    for feature, beta in spec.effects.items():
        if feature in deltas.columns:
            target = target + beta * deltas[feature]
    rate = 1.0 - np.exp(-1.0 / spec.tau_days)
    effect = np.zeros(len(target))
    for i in range(1, len(target)):
        effect[i] = effect[i - 1] + (target.iloc[i] - effect[i - 1]) * rate
    return pd.Series(effect, index=deltas.index)


def confidence_label(sd: float, prior_sd: float) -> str:
    ratio = sd / prior_sd if prior_sd > 0 else 1.0
    if ratio < 0.35:
        return "high"
    if ratio < 0.7:
        return "medium"
    return "low"


def run_filter(
    spec: BiomarkerSpec,
    prior_mean: float,
    prior_sd: float,
    labs: list[tuple[date, float]],
    deltas: pd.DataFrame,
    as_of: date,
) -> FilterResult:
    """Filter from deltas.index[0] to deltas.index[-1]. Days after as_of are forecast."""
    lab_map: dict[pd.Timestamp, list[float]] = {}
    for lab_date, value in labs:
        lab_map.setdefault(pd.Timestamp(lab_date), []).append(float(value))

    effect = effect_path(spec, deltas)
    mean, var = prior_mean, prior_sd**2
    seen_lab = False
    rows = []
    innovations: list[Innovation] = []
    for i, day in enumerate(deltas.index):
        if i > 0:
            step = effect.iloc[i] - effect.iloc[i - 1]
            level = float(inverse(mean, spec.transform))
            if spec.transform == "log":
                mean += float(np.log(max(level + step, 1e-6) / level))
            else:
                mean += step
            var += daily_process_variance(level, spec.annual_drift_cv, spec.transform)

        measured = False
        for value in lab_map.get(day, []):
            y = float(forward(value, spec.transform))
            r = measurement_variance(value, spec.measurement_cv, spec.transform)
            z = (y - mean) / np.sqrt(var + r)
            innovations.append(
                Innovation(day.date(), value, float(inverse(mean, spec.transform)), float(z))
            )
            gain = var / (var + r)
            mean += gain * (y - mean)
            var *= 1.0 - gain
            measured = seen_lab = True

        if measured:
            basis = "measured"
        elif seen_lab:
            basis = "anchored"
        elif abs(effect.iloc[i]) > 0:
            basis = "estimated"
        else:
            basis = "prior"
        sd = float(np.sqrt(var))
        rows.append(
            {
                "date": day.date(),
                "mean": mean,
                "sd": sd,
                "effect": float(effect.iloc[i]),
                "basis": basis,
                "confidence": confidence_label(sd, prior_sd),
                "forecast": day.date() > as_of,
            }
        )

    trajectory = pd.DataFrame(rows)
    quantiles = natural_quantiles(trajectory["mean"], trajectory["sd"], spec.transform)
    low, high = spec.clamp
    for name, values in quantiles.items():
        trajectory[name] = np.clip(values, low, high)
    return FilterResult(trajectory, innovations, prior_mean, prior_sd)
