from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.components.prior_model import QUANTILES

# Population medians and SDs used by the stub prior.
STUB_PRIOR = {
    "non_hdl": (150.0, 40.0),
    "hdl": (50.0, 14.0),
    "hba1c": (5.5, 0.6),
    "egfr": (90.0, 18.0),
    "sbp": (125.0, 16.0),
}
Z = {0.05: -1.645, 0.10: -1.2816, 0.25: -0.6745, 0.50: 0.0, 0.75: 0.6745, 0.90: 1.2816, 0.95: 1.645}


class StubPrior:
    def predict_quantiles(self, profile: pd.DataFrame) -> dict[str, dict[float, float]]:
        return {
            target: {tau: median + Z[tau] * sd for tau in QUANTILES}
            for target, (median, sd) in STUB_PRIOR.items()
        }


@pytest.fixture
def stub_prior() -> StubPrior:
    return StubPrior()


@pytest.fixture
def profile() -> dict:
    return {
        "sex": 1,
        "age": 52,
        "race_eth": 3,
        "height_cm": 178.0,
        "weight_kg": 92.0,
        "smoking": "never",
        "bp_meds": 0,
        "chol_meds": 0,
        "diabetes_dx": "no",
    }


def make_wearables(start: date, n_days: int, change_day: int | None = 45, seed: int = 0) -> pd.DataFrame:
    """Daily wearables. From change_day: +4000 steps and weight falling 0.04 kg/day."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_days):
        active = change_day is not None and i >= change_day
        rows.append(
            {
                "date": start + timedelta(days=i),
                "steps": 5000 + (4000 if active else 0) + rng.normal(0, 500),
                "resting_hr": 70 + rng.normal(0, 1),
                "sleep_h": 7 + rng.normal(0, 0.3),
                "weight_kg": 92 - (0.04 * (i - change_day) if active else 0) + rng.normal(0, 0.2),
            }
        )
    return pd.DataFrame(rows)
