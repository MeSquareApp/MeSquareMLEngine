from datetime import date, timedelta

import pandas as pd

from src.components.personalization.wearable_features import driver_deltas, summarize
from tests.conftest import make_wearables

START = date(2026, 1, 1)


def test_baseline_not_ready_before_30_days():
    summary = summarize(make_wearables(START, 20), START + timedelta(days=19))
    assert not summary.baseline_ready
    deltas = driver_deltas(summary, START, START + timedelta(days=60))
    assert (deltas == 0).all().all()


def test_steps_delta_after_change():
    summary = summarize(make_wearables(START, 120), START + timedelta(days=119))
    assert summary.baseline_ready
    assert summary.baseline_end == START + timedelta(days=29)
    assert abs(summary.latest_delta["steps"] - 4000) < 300
    deltas = driver_deltas(summary, START, START + timedelta(days=150))
    assert (deltas.loc[: pd.Timestamp(summary.baseline_end), "steps"] == 0).all()
    # Steps are in thousands and carried forward into the forecast.
    assert abs(deltas["steps"].iloc[-1] - 4.0) < 0.3


def test_sparse_baseline_feature_is_ignored():
    days = make_wearables(START, 90)
    days.loc[days.index[5:30], "resting_hr"] = None
    summary = summarize(days, START + timedelta(days=89))
    assert summary.baseline["resting_hr"] is None
    assert summary.baseline["steps"] is not None


def test_no_wearables():
    summary = summarize(pd.DataFrame(), START)
    assert summary.first_date is None and not summary.baseline_ready
