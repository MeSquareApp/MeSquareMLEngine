"""Simulated repeat labs to judge the lab stage before real repeat labs exist.

NHANES has one visit per person, so a person's measured value stands in for their
true level. Labs are drawn around it with the published within-person variation in
biomarker_config (the same numbers the lab band uses), and the next lab comes
90-365 days later with drift added. The lab-anchored band and the population band
are both scored on that next lab:

- goal 1: median absolute error of the band centre, lab vs population;
- goal 2: share of next labs inside the 80% band;
- goal 3: lab band width over population band width;
- goal 5: share of "real change" alerts when nothing changed (no drift).

The numbers check that the design is consistent with published variation; they are
not a substitute for real repeat labs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.components.personalization.biomarker_config import BIOMARKERS
from src.components.personalization.distributions import Z_90, drift_sigma, log_sigma
from src.components.personalization.lab_anchor import reference_change

FloatArray = npt.NDArray[np.float64]

NEXT_LAB_DAYS = (90, 365)
GOAL_ERROR_REDUCTION = 0.40
GOAL_COVERAGE = (0.75, 0.85)
GOAL_WIDTH_RATIO = 1.0 / 3.0
GOAL_FALSE_ALERTS = 0.05
SEED = 2026


@dataclass(frozen=True)
class SimulationInput:
    """Held-out people for one analyte.

    Attributes:
        analyte: Canonical analyte name.
        truth: Each person's measured value, used as their true level.
        population: Population (p10, p50, p90) per person, shape (rows, 3).
        weights: Survey weights.
    """

    analyte: str
    truth: FloatArray
    population: FloatArray
    weights: FloatArray


def _weighted_median(values: FloatArray, weights: FloatArray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][np.searchsorted(cumulative, 0.5)])


def simulate_lab_stage(data: SimulationInput) -> dict[str, float | bool]:
    """Score the lab-anchored band against the population band on simulated labs.

    Args:
        data: Held-out people for one analyte.

    Returns:
        Metrics, the pass/fail of goals 1, 2, 3 and 5, and "beats_population" (the
        lab stage's activation rule: lower error, narrower band, coverage in range).
    """
    spec = BIOMARKERS[data.analyte]
    rng = np.random.default_rng(SEED)
    n_rows = len(data.truth)
    measurement = log_sigma(spec.measurement_cv)
    days = rng.uniform(*NEXT_LAB_DAYS, size=n_rows)
    drift = np.array([drift_sigma(spec.annual_drift_cv, day) for day in days])
    first_lab = data.truth * np.exp(rng.normal(0.0, measurement, n_rows))
    next_truth = data.truth * np.exp(rng.normal(0.0, 1.0, n_rows) * drift)
    next_lab = next_truth * np.exp(rng.normal(0.0, measurement, n_rows))

    sigma = np.sqrt(2.0 * measurement**2 + drift**2)
    lab_low, lab_high = (
        first_lab * np.exp(-Z_90 * sigma),
        first_lab * np.exp(Z_90 * sigma),
    )
    pop_low, pop_mid, pop_high = data.population.T
    weights = data.weights

    lab_error = _weighted_median(np.abs(next_lab - first_lab), weights)
    pop_error = _weighted_median(np.abs(next_lab - pop_mid), weights)
    lab_coverage = float(
        np.average((next_lab >= lab_low) & (next_lab <= lab_high), weights=weights)
    )
    pop_coverage = float(
        np.average((next_lab >= pop_low) & (next_lab <= pop_high), weights=weights)
    )
    width_ratio = float(
        np.average(lab_high - lab_low, weights=weights)
        / np.average(pop_high - pop_low, weights=weights)
    )
    stable_repeat = data.truth * np.exp(rng.normal(0.0, measurement, n_rows))
    fall, rise = reference_change(data.analyte)
    change = stable_repeat / first_lab - 1.0
    false_alerts = float(np.average((change < fall) | (change > rise), weights=weights))

    reduction = 1.0 - lab_error / pop_error
    low, high = GOAL_COVERAGE
    return {
        "rows": float(n_rows),
        "lab_median_abs_error": lab_error,
        "population_median_abs_error": pop_error,
        "error_reduction": reduction,
        "lab_coverage_80": lab_coverage,
        "population_coverage_80": pop_coverage,
        "width_ratio": width_ratio,
        "false_alert_rate": false_alerts,
        "min_width_ratio_possible": float(
            math.sqrt(2.0)
            * measurement
            * 2
            * Z_90
            * np.average(data.truth, weights=weights)
            / np.average(pop_high - pop_low, weights=weights)
        ),
        "goal1_error_reduction_met": reduction >= GOAL_ERROR_REDUCTION,
        "goal2_coverage_met": low <= lab_coverage <= high,
        "goal3_width_met": width_ratio <= GOAL_WIDTH_RATIO,
        "goal5_false_alerts_met": false_alerts <= GOAL_FALSE_ALERTS,
        "beats_population": bool(
            lab_error < pop_error and width_ratio < 1.0 and low <= lab_coverage <= high
        ),
    }
