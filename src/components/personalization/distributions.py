"""Move between population quantiles, natural units, and the filter's working scale."""

from __future__ import annotations

import numpy as np

from src.components.personalization.biomarker_config import Transform

# Distance between the 10th and 90th percentiles of a standard normal.
Z_10_90 = 2.5631031310892007
Z_10 = -1.2815515655446004
MIN_SIGMA = 1e-6


def forward(value, transform: Transform):
    return np.log(value) if transform == "log" else np.asarray(value, dtype=float)


def inverse(value, transform: Transform):
    return np.exp(value) if transform == "log" else np.asarray(value, dtype=float)


def normal_from_quantiles(quantiles: dict[float, float], transform: Transform) -> tuple[float, float]:
    """Mean and SD on the working scale from q10, q50, q90."""
    q10, q50, q90 = (forward(max(quantiles[tau], 1e-9), transform) for tau in (0.10, 0.50, 0.90))
    sigma = max(float(q90 - q10) / Z_10_90, MIN_SIGMA)
    return float(q50), sigma


def natural_quantiles(mean, sd, transform: Transform) -> dict[str, np.ndarray]:
    """q10, q50, q90 in natural units for a normal on the working scale."""
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    return {
        "q10": inverse(mean + Z_10 * sd, transform),
        "q50": inverse(mean, transform),
        "q90": inverse(mean - Z_10 * sd, transform),
    }


def measurement_variance(value: float, cv: float, transform: Transform) -> float:
    """Variance of one measurement on the working scale."""
    if transform == "log":
        return float(np.log1p(cv**2))
    return float((cv * value) ** 2)


def daily_process_variance(level: float, annual_cv: float, transform: Transform) -> float:
    """Random-walk variance per day that adds up to annual_cv over a year."""
    if transform == "log":
        return float(np.log1p(annual_cv**2)) / 365.0
    return float((annual_cv * level) ** 2) / 365.0
