from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.components.personalization.biomarker_config import BIOMARKERS, DRIVER_FEATURES
from src.components.personalization.distributions import measurement_variance
from src.components.personalization.kalman import run_filter

START = date(2026, 1, 1)


def zero_deltas(n_days: int) -> pd.DataFrame:
    index = pd.date_range(START, periods=n_days, freq="D")
    return pd.DataFrame(0.0, index=index, columns=list(DRIVER_FEATURES))


def test_no_labs_no_wearables_keeps_prior_mean():
    spec = BIOMARKERS["non_hdl"]
    result = run_filter(spec, 150.0, 40.0, [], zero_deltas(30), START + timedelta(days=29))
    assert np.allclose(result.trajectory["mean"], 150.0)
    assert (result.trajectory["basis"] == "prior").all()
    assert result.trajectory["sd"].is_monotonic_increasing


def test_single_lab_matches_conjugate_update():
    spec = BIOMARKERS["hdl"]
    prior_mean, prior_sd, lab = 50.0, 14.0, 38.0
    result = run_filter(spec, prior_mean, prior_sd, [(START, lab)], zero_deltas(1), START)
    r = measurement_variance(lab, spec.measurement_cv, spec.transform)
    expected_var = 1.0 / (1.0 / prior_sd**2 + 1.0 / r)
    expected_mean = expected_var * (prior_mean / prior_sd**2 + lab / r)
    row = result.trajectory.iloc[0]
    assert row["mean"] == pytest.approx(expected_mean)
    assert row["sd"] == pytest.approx(np.sqrt(expected_var))
    assert row["basis"] == "measured"


def test_lab_shrinks_uncertainty_and_it_regrows():
    spec = BIOMARKERS["hba1c"]
    result = run_filter(spec, np.log(5.5), 0.1, [(START, 6.1)], zero_deltas(365), START)
    sd = result.trajectory["sd"]
    assert sd.iloc[0] < 0.1 * 0.35
    assert sd.iloc[-1] > sd.iloc[0]
    assert (result.trajectory["basis"].iloc[1:] == "anchored").all()


def test_wearable_effect_moves_mean_with_lag():
    spec = BIOMARKERS["sbp"]
    deltas = zero_deltas(200)
    deltas.loc[deltas.index[30]:, "steps"] = 4.0
    result = run_filter(spec, 130.0, 15.0, [], deltas, START + timedelta(days=199))
    target = spec.effects["steps"] * 4.0
    effect = result.trajectory["effect"]
    assert effect.iloc[29] == 0
    assert 0 > effect.iloc[40] > target
    assert effect.iloc[-1] == pytest.approx(target, rel=0.01)
    assert result.trajectory["basis"].iloc[-1] == "estimated"


def test_innovation_z_score():
    spec = BIOMARKERS["non_hdl"]
    result = run_filter(spec, 150.0, 40.0, [(START, 150.0)], zero_deltas(1), START)
    assert result.innovations[0].z == pytest.approx(0.0)
