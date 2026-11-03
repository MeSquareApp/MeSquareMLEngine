"""Move between population quantiles, natural units, and the filter's working scale."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.components.personalization.biomarker_config import Transform

FloatArray = npt.NDArray[np.float64]

# Distance between the 10th and 90th percentiles of a standard normal.
Z_10_90 = 2.5631031310892007
Z_10 = -1.2815515655446004
MIN_SIGMA = 1e-6
DAYS_PER_YEAR = 365.0


def forward(value: npt.ArrayLike, transform: Transform) -> FloatArray:
    """Map natural-unit values onto the working scale.

    Args:
        value: Values in the biomarker's natural unit. Must be positive for "log".
        transform: "log" or "identity".

    Returns:
        Values on the working scale.
    """
    values = np.asarray(value, dtype=np.float64)
    return np.log(values) if transform == "log" else values


def inverse(value: npt.ArrayLike, transform: Transform) -> FloatArray:
    """Map working-scale values back to natural units.

    Args:
        value: Values on the working scale.
        transform: "log" or "identity".

    Returns:
        Values in the biomarker's natural unit.
    """
    values = np.asarray(value, dtype=np.float64)
    return np.exp(values) if transform == "log" else values


def normal_from_quantiles(
    quantiles: dict[float, float], transform: Transform
) -> tuple[float, float]:
    """Fit a normal on the working scale from population p10, p50, and p90.

    Args:
        quantiles: Natural-unit values keyed by tau. Must include 0.10, 0.50, 0.90.
        transform: Working-scale transform of the biomarker.

    Returns:
        Mean and standard deviation on the working scale.

    Raises:
        KeyError: If any of the three required quantiles is missing.
    """
    q10, q50, q90 = forward(
        [max(quantiles[tau], 1e-9) for tau in (0.10, 0.50, 0.90)], transform
    )
    sigma = max(float(q90 - q10) / Z_10_90, MIN_SIGMA)
    return float(q50), sigma


def natural_quantiles(
    mean: FloatArray, sd: FloatArray, transform: Transform
) -> dict[str, FloatArray]:
    """Natural-unit p10, p50, and p90 of normals on the working scale.

    Args:
        mean: Working-scale means.
        sd: Working-scale standard deviations, same shape as mean.
        transform: Working-scale transform of the biomarker.

    Returns:
        Arrays keyed "q10", "q50", and "q90".
    """
    return {
        "q10": inverse(mean + Z_10 * sd, transform),
        "q50": inverse(mean, transform),
        "q90": inverse(mean - Z_10 * sd, transform),
    }


def measurement_variance(value: float, cv: float, transform: Transform) -> float:
    """Variance of one lab measurement on the working scale.

    Args:
        value: Measured value in natural units.
        cv: Combined analytical and within-person coefficient of variation.
        transform: Working-scale transform of the biomarker.

    Returns:
        Measurement variance on the working scale.
    """
    if transform == "log":
        return float(np.log1p(cv**2))
    return float((cv * value) ** 2)


def daily_process_variance(
    level: float, annual_cv: float, transform: Transform
) -> float:
    """Random-walk variance per day that adds up to annual_cv over one year.

    Args:
        level: Current level in natural units. Used on the identity scale only.
        annual_cv: Within-person drift over a year as a coefficient of variation.
        transform: Working-scale transform of the biomarker.

    Returns:
        Daily process variance on the working scale.
    """
    if transform == "log":
        return float(np.log1p(annual_cv**2)) / DAYS_PER_YEAR
    return float((annual_cv * level) ** 2) / DAYS_PER_YEAR
