from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from src.components.personalization.wearable_windows import (
    FEATURE_NAMES,
    feature_vector,
    feature_windows,
    summarize_window,
)
from src.components.record_ingestion import Measurement

END = date(2026, 4, 30)


def daily(n_days: int, slope: float = 0.0, skip_every: int = 0) -> dict[date, float]:
    return {
        END - timedelta(days=offset): 60.0 - slope * offset
        for offset in range(n_days)
        if not (skip_every and offset % skip_every == 0)
    }


def test_window_statistics_and_theil_sen_slope():
    # Arrange: rises 0.1 bpm per day for 30 days, with one outlier.
    values = daily(30, slope=0.1)
    values[END - timedelta(days=5)] = 200.0

    # Act
    window = summarize_window(values, "resting_heart_rate", 30, END)

    # Assert
    assert window.sufficient
    assert window.coverage == 1.0
    assert window.slope_per_day == pytest.approx(0.1, abs=1e-6)


def test_window_below_coverage_is_insufficient():
    # Arrange: half the days are missing.
    values = daily(30, skip_every=2)

    # Act
    window = summarize_window(values, "resting_heart_rate", 30, END)

    # Assert
    assert window.coverage == 0.5
    assert not window.sufficient
    assert window.mean is None


def test_feature_vector_has_every_feature_with_nan_for_insufficient():
    # Arrange
    wearables = [
        Measurement(
            f"w{offset}",
            "resting_heart_rate",
            60.0,
            datetime.combine(END - timedelta(days=offset), datetime.min.time(), UTC),
            "wearable_apple_health",
        )
        for offset in range(40)
    ]

    # Act
    features = feature_vector(feature_windows(wearables, END))

    # Assert
    assert tuple(features) == FEATURE_NAMES
    assert features["resting_heart_rate_30d_mean"] == 60.0
    assert np.isnan(features["resting_heart_rate_90d_mean"])
    assert np.isnan(features["hrv_rmssd_30d_mean"])
