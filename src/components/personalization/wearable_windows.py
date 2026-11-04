"""Wearable feature windows: 30- and 90-day summaries of each daily metric.

Each window ending at as_of gets the mean, SD, Theil-Sen slope per day and the share
of days with data. A window is used only with at least MIN_COVERAGE of its days
covered; otherwise it is marked insufficient and its statistics are left out.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from scipy.stats import theilslopes

from src.components.record_ingestion import Measurement

WEARABLE_METRICS = ("resting_heart_rate", "hrv_rmssd", "vo2_max", "weight")
WINDOW_DAYS = (30, 90)
MIN_COVERAGE = 0.7
WINDOW_STATS = ("mean", "sd", "slope")


@dataclass(frozen=True)
class WindowSummary:
    """One metric over one window ending at as_of.

    Attributes:
        metric: Wearable metric name.
        days: Window length in days.
        coverage: Share of days in the window with a daily value.
        sufficient: Whether coverage is at least MIN_COVERAGE.
        mean: Mean of the daily values, when sufficient.
        sd: Standard deviation of the daily values, when sufficient.
        slope_per_day: Theil-Sen slope in units per day, when sufficient.
    """

    metric: str
    days: int
    coverage: float
    sufficient: bool
    mean: float | None = None
    sd: float | None = None
    slope_per_day: float | None = None


def daily_values(
    wearables: Sequence[Measurement], metric: str, as_of: date
) -> dict[date, float]:
    """One value per day for a metric, up to as_of. Several on a day are averaged.

    Args:
        wearables: Accepted wearable measurements.
        metric: Wearable metric name.
        as_of: Last day to include.

    Returns:
        Daily value keyed by day.
    """
    by_day: defaultdict[date, list[float]] = defaultdict(list)
    for item in wearables:
        if item.analyte == metric and item.day <= as_of:
            by_day[item.day].append(item.value)
    return {day: float(np.mean(values)) for day, values in by_day.items()}


def summarize_window(
    daily: dict[date, float], metric: str, days: int, as_of: date
) -> WindowSummary:
    """Summarize one metric over the window of days ending at as_of.

    Args:
        daily: Daily values keyed by day.
        metric: Wearable metric name.
        days: Window length.
        as_of: Last day of the window.

    Returns:
        The window summary.
    """
    start = as_of - timedelta(days=days - 1)
    points = sorted(
        (day, value) for day, value in daily.items() if start <= day <= as_of
    )
    coverage = len(points) / days
    if coverage < MIN_COVERAGE:
        return WindowSummary(metric, days, coverage, sufficient=False)
    offsets = np.array([(day - start).days for day, _ in points], dtype=np.float64)
    values = np.array([value for _, value in points])
    slope = float(theilslopes(values, offsets).slope)
    return WindowSummary(
        metric,
        days,
        coverage,
        sufficient=True,
        mean=float(values.mean()),
        sd=float(values.std(ddof=1)),
        slope_per_day=slope,
    )


def feature_windows(
    wearables: Sequence[Measurement], as_of: date
) -> dict[str, WindowSummary]:
    """Every metric x window summary ending at as_of.

    Args:
        wearables: Accepted wearable measurements.
        as_of: Last day of every window.

    Returns:
        Summaries keyed "<metric>_<days>d".
    """
    windows: dict[str, WindowSummary] = {}
    for metric in WEARABLE_METRICS:
        daily = daily_values(wearables, metric, as_of)
        for days in WINDOW_DAYS:
            windows[f"{metric}_{days}d"] = summarize_window(daily, metric, days, as_of)
    return windows


def feature_vector(windows: dict[str, WindowSummary]) -> dict[str, float]:
    """Flatten window summaries into model features. Insufficient windows are NaN.

    Args:
        windows: Output of feature_windows().

    Returns:
        "<metric>_<days>d_<stat>" -> value, for every stat in WINDOW_STATS.
    """
    features: dict[str, float] = {}
    for key, window in windows.items():
        values = (window.mean, window.sd, window.slope_per_day)
        for stat, value in zip(WINDOW_STATS, values, strict=True):
            features[f"{key}_{stat}"] = np.nan if value is None else value
    return features


FEATURE_NAMES = tuple(
    f"{metric}_{days}d_{stat}"
    for metric in WEARABLE_METRICS
    for days in WINDOW_DAYS
    for stat in WINDOW_STATS
)
