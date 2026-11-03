"""Turn end-of-day wearable aggregates into changes against the user's own baseline.

The baseline is the mean over the first BASELINE_DAYS calendar days of wearable data.
After that, each driver's change is its trailing BASELINE_DAYS mean minus the baseline.
A window counts only with at least MIN_VALID_DAYS days of data. Through gaps and into
the forecast the last known change is carried forward, which assumes the behaviour
persists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import cast

import polars as pl

from src.components.personalization.biomarker_config import (
    BASELINE_DAYS,
    DRIVER_FEATURES,
    FEATURE_SCALE,
    MIN_VALID_DAYS,
    WEARABLE_FEATURES,
)

WearableRecord = Mapping[str, date | float | None]

SHORT_WINDOW_DAYS = 7
SHORT_WINDOW_MIN_DAYS = 4
DAILY_SCHEMA = {"date": pl.Date, **{name: pl.Float64 for name in WEARABLE_FEATURES}}


@dataclass(frozen=True)
class WearableSummary:
    """Baseline and trailing-window view of a user's wearable history.

    Attributes:
        first_date: First day with any wearable record, or None without data.
        baseline_end: Last day of the baseline window, or None without data.
        baseline_ready: Whether the baseline window has fully elapsed by as_of.
        baseline: Baseline mean per feature. None when too few valid days.
        latest_delta: Latest trailing mean minus baseline, in natural units.
        rolling: One row per calendar day: date, <feature>_7d, <feature>_30d.
    """

    first_date: date | None = None
    baseline_end: date | None = None
    baseline_ready: bool = False
    baseline: dict[str, float | None] = field(default_factory=dict)
    latest_delta: dict[str, float | None] = field(default_factory=dict)
    rolling: pl.DataFrame = field(default_factory=pl.DataFrame)


def daily_frame(wearable_days: Sequence[WearableRecord]) -> pl.DataFrame:
    """Build one row per calendar day from the first to the last record.

    Days without a record get nulls. Several records on one day are averaged.

    Args:
        wearable_days: End-of-day records with a "date" key and optional features.

    Returns:
        Frame with "date" and every feature in WEARABLE_FEATURES.
    """
    records = [
        {name: record.get(name) for name in DAILY_SCHEMA} for record in wearable_days
    ]
    frame = pl.DataFrame(records, schema=DAILY_SCHEMA)
    if frame.is_empty():
        return frame
    frame = frame.group_by("date").agg(pl.col(WEARABLE_FEATURES).mean())
    first = cast(date, frame["date"].min())
    last = cast(date, frame["date"].max())
    return calendar_frame(first, last).join(frame, on="date", how="left")


def calendar_frame(start: date, end: date) -> pl.DataFrame:
    """One row per calendar day from start to end inclusive.

    Args:
        start: First day.
        end: Last day.

    Returns:
        Frame with a single "date" column.
    """
    n_days = (end - start).days + 1
    days = [start + timedelta(days=offset) for offset in range(n_days)]
    return pl.DataFrame({"date": days}, schema={"date": pl.Date})


def summarize(wearable_days: Sequence[WearableRecord], as_of: date) -> WearableSummary:
    """Compute the user's baseline, trailing means, and latest change up to as_of.

    Args:
        wearable_days: End-of-day records with a "date" key and optional features.
        as_of: Last day to use. Later records are ignored.

    Returns:
        The wearable summary. Empty when there is no data on or before as_of.
    """
    daily = daily_frame(wearable_days).filter(pl.col("date") <= as_of)
    if daily.is_empty():
        return WearableSummary()

    first_date = cast(date, daily["date"].min())
    baseline_end = first_date + timedelta(days=BASELINE_DAYS - 1)
    ready = as_of >= baseline_end
    baseline = _baseline_means(daily, baseline_end) if ready else {}

    rolling = daily.select(
        "date",
        *(
            pl.col(name)
            .rolling_mean(SHORT_WINDOW_DAYS, min_samples=SHORT_WINDOW_MIN_DAYS)
            .alias(f"{name}_7d")
            for name in WEARABLE_FEATURES
        ),
        *(
            pl.col(name)
            .rolling_mean(BASELINE_DAYS, min_samples=MIN_VALID_DAYS)
            .alias(f"{name}_30d")
            for name in WEARABLE_FEATURES
        ),
    )
    latest = rolling.select(
        pl.col(f"{name}_30d").drop_nulls().last().alias(name)
        for name in WEARABLE_FEATURES
    ).row(0, named=True)
    latest_delta = {
        name: None
        if baseline.get(name) is None or latest[name] is None
        else latest[name] - baseline[name]
        for name in WEARABLE_FEATURES
    }
    return WearableSummary(
        first_date=first_date,
        baseline_end=baseline_end,
        baseline_ready=ready,
        baseline={name: baseline.get(name) for name in WEARABLE_FEATURES},
        latest_delta=latest_delta,
        rolling=rolling,
    )


def _baseline_means(daily: pl.DataFrame, baseline_end: date) -> dict[str, float | None]:
    """Mean of each feature over the baseline window, if it has enough valid days."""
    window = daily.filter(pl.col("date") <= baseline_end)
    row = window.select(
        pl.when(pl.col(name).count() >= MIN_VALID_DAYS)
        .then(pl.col(name).mean())
        .alias(name)
        for name in WEARABLE_FEATURES
    ).row(0, named=True)
    return {name: row[name] for name in WEARABLE_FEATURES}


def driver_deltas(summary: WearableSummary, start: date, end: date) -> pl.DataFrame:
    """Daily driver changes over [start, end] in model units (steps in thousands).

    Zero until the baseline window closes, then carried forward after the last valid
    trailing window.

    Args:
        summary: Output of summarize().
        start: First day of the filter timeline.
        end: Last day of the filter timeline, including the forecast horizon.

    Returns:
        Frame with "date" and one column per driver in DRIVER_FEATURES.
    """
    calendar = calendar_frame(start, end)
    baseline_end = summary.baseline_end
    usable = [
        name for name in DRIVER_FEATURES if summary.baseline.get(name) is not None
    ]
    if not summary.baseline_ready or baseline_end is None or not usable:
        return calendar.with_columns(
            pl.lit(0.0).alias(name) for name in DRIVER_FEATURES
        )

    changes = summary.rolling.filter(pl.col("date") > baseline_end).select(
        "date",
        *(
            (
                (pl.col(f"{name}_30d") - summary.baseline[name])
                / FEATURE_SCALE.get(name, 1.0)
            ).alias(name)
            for name in usable
        ),
    )
    after_baseline = pl.col("date") > baseline_end
    return calendar.join(changes, on="date", how="left").select(
        "date",
        *(
            pl.when(after_baseline)
            .then(pl.col(name).forward_fill().fill_null(0.0))
            .otherwise(0.0)
            .alias(name)
            if name in usable
            else pl.lit(0.0).alias(name)
            for name in DRIVER_FEATURES
        ),
    )
