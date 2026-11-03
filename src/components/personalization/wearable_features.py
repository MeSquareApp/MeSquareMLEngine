"""Turn end-of-day wearable aggregates into changes against the user's own baseline.

Days the wrist device was barely worn are dropped: with wear_hours below
MIN_WEAR_HOURS, or without wear_hours and fewer than MIN_STEPS_WORN steps. Weight comes
from a scale and is kept. The baseline is the mean over the first BASELINE_DAYS
calendar days of wearable data. After that, each driver's change is its trailing
BASELINE_DAYS mean minus the baseline. A window counts only with at least
MIN_VALID_DAYS valid days. Through a gap the last change is carried forward, fading
with a STALE_HALF_LIFE_DAYS half-life once the gap passes STALE_AFTER_DAYS. Into the
forecast after as_of the change is held, which assumes the behaviour persists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal, cast

import polars as pl

from src.components.personalization.biomarker_config import (
    BASELINE_DAYS,
    DRIVER_FEATURES,
    FEATURE_SCALE,
    MIN_STEPS_WORN,
    MIN_VALID_DAYS,
    MIN_WEAR_HOURS,
    STALE_AFTER_DAYS,
    STALE_HALF_LIFE_DAYS,
    WEARABLE_FEATURES,
    WRIST_FEATURES,
)

WearableRecord = Mapping[str, date | float | None]
WearableStatus = Literal["no_data", "building_baseline", "ok", "sparse", "stale"]

SHORT_WINDOW_DAYS = 7
SHORT_WINDOW_MIN_DAYS = 4
DAILY_SCHEMA = {
    "date": pl.Date,
    "wear_hours": pl.Float64,
    **{name: pl.Float64 for name in WEARABLE_FEATURES},
}


@dataclass(frozen=True)
class WearableSummary:
    """Baseline and trailing-window view of a user's wearable history.

    Attributes:
        as_of: Last day considered.
        first_date: First day with any wearable record, or None without data.
        baseline_end: Last day of the baseline window, or None without data.
        baseline_ready: Whether the baseline window has fully elapsed by as_of.
        baseline: Baseline mean per feature. None when too few valid days.
        latest_delta: Latest trailing mean minus baseline, in natural units.
        rolling: One row per calendar day: date, <feature>_7d, <feature>_30d.
        valid_days_last_30: Days in the last BASELINE_DAYS with valid wrist data.
        last_valid_date: Latest day with valid wrist data, or None.
        status: "no_data", "building_baseline", "ok", "sparse", or "stale".
    """

    as_of: date | None = None
    first_date: date | None = None
    baseline_end: date | None = None
    baseline_ready: bool = False
    baseline: dict[str, float | None] = field(default_factory=dict)
    latest_delta: dict[str, float | None] = field(default_factory=dict)
    rolling: pl.DataFrame = field(default_factory=pl.DataFrame)
    valid_days_last_30: int = 0
    last_valid_date: date | None = None
    status: WearableStatus = "no_data"


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


def daily_frame(wearable_days: Sequence[WearableRecord]) -> pl.DataFrame:
    """Build one row per calendar day, with partial-wear wrist data removed.

    Days without a record get nulls. Several records on one day are averaged. On
    days the device was barely worn, every wrist feature becomes null.

    Args:
        wearable_days: End-of-day records with a "date" key, optional features,
            and an optional "wear_hours".

    Returns:
        Frame with "date" and every feature in WEARABLE_FEATURES.
    """
    records = [
        {name: record.get(name) for name in DAILY_SCHEMA} for record in wearable_days
    ]
    frame = pl.DataFrame(records, schema=DAILY_SCHEMA)
    if frame.is_empty():
        return frame.drop("wear_hours")
    frame = frame.group_by("date").agg(pl.exclude("date").mean())
    worn = (
        pl.when(pl.col("wear_hours").is_not_null())
        .then(pl.col("wear_hours") >= MIN_WEAR_HOURS)
        .otherwise(pl.col("steps").is_null() | (pl.col("steps") >= MIN_STEPS_WORN))
    )
    frame = frame.with_columns(
        pl.when(worn).then(pl.col(name)).alias(name) for name in WRIST_FEATURES
    ).drop("wear_hours")
    first = cast(date, frame["date"].min())
    last = cast(date, frame["date"].max())
    return calendar_frame(first, last).join(frame, on="date", how="left")


def summarize(wearable_days: Sequence[WearableRecord], as_of: date) -> WearableSummary:
    """Compute the user's baseline, trailing means, latest change, and data status.

    Args:
        wearable_days: End-of-day records with a "date" key and optional features.
        as_of: Last day to use. Later records are ignored.

    Returns:
        The wearable summary. Status "no_data" when nothing is on or before as_of.
    """
    daily = daily_frame(wearable_days).filter(pl.col("date") <= as_of)
    if daily.is_empty():
        return WearableSummary(as_of=as_of)

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
    valid_days, last_valid_date = _wear_coverage(daily, as_of)
    return WearableSummary(
        as_of=as_of,
        first_date=first_date,
        baseline_end=baseline_end,
        baseline_ready=ready,
        baseline={name: baseline.get(name) for name in WEARABLE_FEATURES},
        latest_delta=latest_delta,
        rolling=rolling,
        valid_days_last_30=valid_days,
        last_valid_date=last_valid_date,
        status=_status(ready, valid_days, last_valid_date, as_of),
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


def _wear_coverage(daily: pl.DataFrame, as_of: date) -> tuple[int, date | None]:
    """Valid wrist days in the last BASELINE_DAYS, and the latest valid day."""
    valid = daily.filter(pl.any_horizontal(pl.col(WRIST_FEATURES).is_not_null()))
    if valid.is_empty():
        return 0, None
    window_start = as_of - timedelta(days=BASELINE_DAYS - 1)
    recent = valid.filter(pl.col("date") >= window_start).height
    return recent, cast(date, valid["date"].max())


def _status(
    ready: bool, valid_days: int, last_valid_date: date | None, as_of: date
) -> WearableStatus:
    """Data status shown to the user. The worst condition wins."""
    if last_valid_date is None:
        return "no_data"
    if (as_of - last_valid_date).days > STALE_AFTER_DAYS:
        return "stale"
    if not ready:
        return "building_baseline"
    return "ok" if valid_days >= MIN_VALID_DAYS else "sparse"


def driver_deltas(summary: WearableSummary, start: date, end: date) -> pl.DataFrame:
    """Daily driver changes over [start, end] in model units (steps in thousands).

    Zero until the baseline window closes. Through a gap the last valid change is
    carried forward and fades once the gap passes STALE_AFTER_DAYS. After as_of the
    change is held at its as_of value.

    Args:
        summary: Output of summarize().
        start: First day of the filter timeline.
        end: Last day of the filter timeline, including the forecast horizon.

    Returns:
        Frame with "date" and one column per driver in DRIVER_FEATURES.
    """
    calendar = calendar_frame(start, end)
    baseline_end, as_of = summary.baseline_end, summary.as_of
    usable = [
        name for name in DRIVER_FEATURES if summary.baseline.get(name) is not None
    ]
    if (
        not summary.baseline_ready
        or baseline_end is None
        or as_of is None
        or not usable
    ):
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
    joined = calendar.join(changes, on="date", how="left")
    observed_until = pl.min_horizontal(pl.col("date"), pl.lit(as_of))
    return joined.select(
        "date",
        *(
            pl.when(pl.col("date") > baseline_end)
            .then(_faded_change(name, observed_until))
            .otherwise(0.0)
            .alias(name)
            if name in usable
            else pl.lit(0.0).alias(name)
            for name in DRIVER_FEATURES
        ),
    )


def _faded_change(name: str, observed_until: pl.Expr) -> pl.Expr:
    """Last valid change of one driver, faded by how stale it was at observed_until."""
    last_valid = pl.when(pl.col(name).is_not_null()).then(pl.col("date")).forward_fill()
    gap_days = (observed_until - last_valid).dt.total_days()
    excess = (gap_days - STALE_AFTER_DAYS).clip(lower_bound=0)
    fade = pl.lit(0.5).pow(excess / STALE_HALF_LIFE_DAYS)
    return (pl.col(name).forward_fill() * fade).fill_null(0.0)
