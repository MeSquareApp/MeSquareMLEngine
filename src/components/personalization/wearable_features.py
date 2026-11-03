"""Turn end-of-day wearable aggregates into behaviour changes against the user's own baseline.

The baseline is the mean over the first BASELINE_DAYS calendar days of wearable data. After
that, each driver's change is its trailing BASELINE_DAYS mean minus the baseline. A window
counts only with at least MIN_VALID_DAYS days of data. Through gaps and into the forecast,
the last known change is carried forward, which assumes the behaviour persists.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.components.personalization.biomarker_config import (
    BASELINE_DAYS,
    DRIVER_FEATURES,
    FEATURE_SCALE,
    MIN_VALID_DAYS,
    WEARABLE_FEATURES,
)


@dataclass(frozen=True)
class WearableSummary:
    first_date: date | None
    baseline_end: date | None
    baseline_ready: bool
    baseline: dict[str, float | None]
    # Latest trailing-window mean minus baseline, in natural units (steps per day, not thousands).
    latest_delta: dict[str, float | None]
    # Index: every calendar day of wearable data. Columns: <feature>_7d, <feature>_30d.
    rolling: pd.DataFrame


def daily_frame(wearable_days: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day from first to last record. Missing days are NaN."""
    if wearable_days.empty:
        return pd.DataFrame(columns=list(WEARABLE_FEATURES), index=pd.DatetimeIndex([]), dtype=float)
    frame = wearable_days.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.groupby("date").mean(numeric_only=True)
    index = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    frame = frame.reindex(index)
    for column in WEARABLE_FEATURES:
        if column not in frame.columns:
            frame[column] = float("nan")
    return frame[list(WEARABLE_FEATURES)].astype(float)


def summarize(wearable_days: pd.DataFrame, as_of: date) -> WearableSummary:
    daily = daily_frame(wearable_days)
    daily = daily[daily.index <= pd.Timestamp(as_of)]
    if daily.empty:
        return WearableSummary(None, None, False, {}, {}, pd.DataFrame())

    first = daily.index.min()
    baseline_end = first + timedelta(days=BASELINE_DAYS - 1)
    ready = pd.Timestamp(as_of) >= baseline_end
    window = daily[daily.index <= baseline_end]

    baseline: dict[str, float | None] = {}
    for column in WEARABLE_FEATURES:
        values = window[column].dropna()
        baseline[column] = float(values.mean()) if ready and len(values) >= MIN_VALID_DAYS else None

    rolling = pd.DataFrame(index=daily.index)
    for column in WEARABLE_FEATURES:
        rolling[f"{column}_7d"] = daily[column].rolling(7, min_periods=4).mean()
        rolling[f"{column}_30d"] = (
            daily[column].rolling(BASELINE_DAYS, min_periods=MIN_VALID_DAYS).mean()
        )

    latest_delta: dict[str, float | None] = {}
    for column in WEARABLE_FEATURES:
        recent = rolling[f"{column}_30d"].dropna()
        if baseline[column] is None or recent.empty:
            latest_delta[column] = None
        else:
            latest_delta[column] = float(recent.iloc[-1] - baseline[column])

    return WearableSummary(
        first_date=first.date(),
        baseline_end=baseline_end.date(),
        baseline_ready=bool(ready),
        baseline=baseline,
        latest_delta=latest_delta,
        rolling=rolling,
    )


def driver_deltas(summary: WearableSummary, start: date, end: date) -> pd.DataFrame:
    """Daily driver changes over [start, end] in model units (steps in thousands).

    The value is zero until the baseline window closes and is carried forward after the last
    valid window.
    """
    index = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    deltas = pd.DataFrame(0.0, index=index, columns=list(DRIVER_FEATURES))
    if summary.baseline_end is None or not summary.baseline_ready:
        return deltas

    after_baseline = summary.rolling.index > pd.Timestamp(summary.baseline_end)
    for column in DRIVER_FEATURES:
        base = summary.baseline.get(column)
        if base is None:
            continue
        series = summary.rolling.loc[after_baseline, f"{column}_30d"] - base
        series = series.reindex(index).ffill().fillna(0.0)
        # Before the baseline closes there is no change to measure.
        series[index <= pd.Timestamp(summary.baseline_end)] = 0.0
        deltas[column] = series / FEATURE_SCALE.get(column, 1.0)
    return deltas
