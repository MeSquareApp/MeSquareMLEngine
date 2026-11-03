from datetime import date, timedelta

import polars as pl

from src.components.personalization.wearable_features import (
    WearableRecord,
    daily_frame,
    driver_deltas,
    summarize,
)
from tests.conftest import STEP_INCREASE, make_wearables

START = date(2026, 1, 1)


def test_daily_frame_fills_missing_days_and_averages_duplicates():
    # Arrange
    records: list[WearableRecord] = [
        {"date": START, "steps": 1000.0},
        {"date": START, "steps": 3000.0},
        {"date": START + timedelta(days=2), "steps": 5000.0},
    ]

    # Act
    frame = daily_frame(records)

    # Assert
    assert frame["date"].to_list() == [START + timedelta(days=i) for i in range(3)]
    assert frame["steps"].to_list() == [2000.0, None, 5000.0]


def test_baseline_not_ready_before_30_days():
    # Arrange
    records = make_wearables(START, 20)

    # Act
    summary = summarize(records, START + timedelta(days=19))
    deltas = driver_deltas(summary, START, START + timedelta(days=60))

    # Assert
    assert not summary.baseline_ready
    assert deltas.drop("date").sum().row(0) == (0.0,) * (deltas.width - 1)


def test_steps_delta_after_change_is_carried_into_forecast():
    # Arrange
    records = make_wearables(START, 120)

    # Act
    summary = summarize(records, START + timedelta(days=119))
    deltas = driver_deltas(summary, START, START + timedelta(days=150))

    # Assert
    assert summary.baseline_ready
    assert summary.baseline_end == START + timedelta(days=29)
    steps_delta = summary.latest_delta["steps"]
    assert steps_delta is not None
    assert abs(steps_delta - STEP_INCREASE) < 300
    before = deltas.filter(pl.col("date") <= summary.baseline_end)["steps"]
    assert (before == 0).all()
    assert abs(deltas["steps"][-1] - STEP_INCREASE / 1000) < 0.3


def test_sparse_baseline_feature_is_ignored():
    # Arrange
    records = make_wearables(START, 90)
    for record in records[5:30]:
        record["resting_hr"] = None

    # Act
    summary = summarize(records, START + timedelta(days=89))

    # Assert
    assert summary.baseline["resting_hr"] is None
    assert summary.baseline["steps"] is not None


def test_no_wearables_gives_empty_summary():
    # Arrange
    records: list[WearableRecord] = []

    # Act
    summary = summarize(records, START)

    # Assert
    assert summary.first_date is None
    assert not summary.baseline_ready
