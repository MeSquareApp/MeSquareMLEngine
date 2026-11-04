"""Log-normal bands: from quantiles, from a lab, and how they widen with time."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# 90th percentile of a standard normal. p10 and p90 sit this many SDs from p50.
Z_90 = 1.2815515655446004
# Two-sided 95% critical value, for the reference change value.
Z_95 = 1.959963984540054
MIN_SIGMA = 1e-6
DAYS_PER_YEAR = 365.0


def log_sigma(cv: float) -> float:
    """SD on the log scale of a log-normal with the given coefficient of variation.

    Args:
        cv: Coefficient of variation, for example 0.07.

    Returns:
        sqrt(ln(1 + cv^2)).
    """
    return math.sqrt(math.log1p(cv**2))


def drift_sigma(annual_cv: float, days: float) -> float:
    """Log-scale SD of within-person drift accumulated over a number of days.

    Args:
        annual_cv: Drift over one year as a coefficient of variation.
        days: Elapsed days. Negative values count as zero.

    Returns:
        The SD, growing with the square root of time.
    """
    return log_sigma(annual_cv) * math.sqrt(max(days, 0.0) / DAYS_PER_YEAR)


def normal_from_quantiles(quantiles: dict[float, float]) -> tuple[float, float]:
    """Log-scale mean and SD from p10, p50, p90 in natural units.

    Args:
        quantiles: Values keyed by tau. Must include 0.10, 0.50, 0.90.

    Returns:
        (mean, sd) on the log scale.

    Raises:
        KeyError: If a required quantile is missing.
    """
    q10, q50, q90 = (math.log(max(quantiles[tau], 1e-9)) for tau in (0.10, 0.50, 0.90))
    return q50, max((q90 - q10) / (2 * Z_90), MIN_SIGMA)


def band(center: float, sigma: float) -> tuple[float, float, float]:
    """p10, p50, p90 of a log-normal centred on center.

    Args:
        center: Median in natural units.
        sigma: Log-scale SD.

    Returns:
        (p10, p50, p90).
    """
    return (
        center * math.exp(-Z_90 * sigma),
        center,
        center * math.exp(Z_90 * sigma),
    )
