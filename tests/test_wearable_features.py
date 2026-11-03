from datetime import date, timedelta

import polars as pl
import pytest

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


def steady_days(
    n_days: int, steps: float = 8000.0
) -> list[dict[str, date | float | None]]:
    return [
        {"date": START + timedelta(days=i), "steps": steps, "resting_hr": 65.0}
        for i in range(n_days)
    ]


def test_short_wear_days_are_dropped_but_weight_is_kept():
    # Arrange
    records: list[WearableRecord] = [
        {"date": START, "steps": 1200.0, "wear_hours": 4.0, "weight_kg": 90.0},
        {"date": START + timedelta(days=1), "steps": 900.0},
        {"date": START + timedelta(days=2), "steps": 900.0, "wear_hours": 15.0},
    ]

    # Act
    frame = daily_frame(records)

    # Assert
    assert frame["steps"].to_list() == [None, None, 900.0]
    assert frame["weight_kg"].to_list() == [90.0, None, None]


def test_partial_wear_days_do_not_create_a_fake_activity_drop():
    # Arrange: same behaviour throughout, but 10 short-wear days late in the history.
    records = steady_days(90)
    for record in records[70:80]:
        record.update({"steps": 1200.0, "wear_hours": 5.0})

    # Act
    summary = summarize(records, START + timedelta(days=89))

    # Assert
    steps_delta = summary.latest_delta["steps"]
    assert steps_delta is not None
    assert abs(steps_delta) < 1.0


def test_change_fades_after_a_long_gap_but_holds_into_the_forecast():
    # Arrange: +4000 steps from day 30, then no data after day 89.
    records = steady_days(30, 5000.0) + [
        {"date": START + timedelta(days=i), "steps": 9000.0} for i in range(30, 90)
    ]
    long_gap = START + timedelta(days=200)

    # Act
    fresh = summarize(records, START + timedelta(days=89))
    stale = summarize(records, long_gap)
    fresh_deltas = driver_deltas(fresh, START, START + timedelta(days=150))
    stale_deltas = driver_deltas(stale, START, long_gap)

    # Assert
    assert fresh_deltas["steps"][-1] == pytest.approx(fresh_deltas["steps"][89])
    assert stale_deltas["steps"][-1] < 0.2 * stale_deltas["steps"][89]
    assert stale.status == "stale"


@pytest.mark.parametrize(
    ("n_days", "keep_every", "expected"),
    [(20, 1, "building_baseline"), (60, 1, "ok"), (60, 3, "sparse")],
)
def test_wearable_status(n_days, keep_every, expected):
    # Arrange
    records = steady_days(n_days)[::keep_every]

    # Act
    summary = summarize(records, START + timedelta(days=n_days - 1))

    # Assert
    assert summary.status == expected
