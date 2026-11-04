from datetime import UTC, date, datetime, timedelta

import pytest

from src.components.personalization.lab_anchor import (
    TreatmentChanges,
    anchoring_labs,
    change_flag,
    lab_band,
    reference_change,
)
from src.components.record_ingestion import Measurement

DAY = date(2026, 1, 1)


def measurement(analyte: str, value: float, day: date = DAY) -> Measurement:
    return Measurement(
        record_id=f"{analyte}-{day}",
        analyte=analyte,
        value=value,
        measured_at=datetime(day.year, day.month, day.day, tzinfo=UTC),
        method="blood_test_lab_report_upload",
    )


def test_lab_band_is_centred_on_the_lab_and_widens_with_time():
    # Arrange
    lab = measurement("hdl", 50.0)

    # Act
    today = lab_band(lab, DAY)
    later = lab_band(lab, DAY + timedelta(days=365))

    # Assert
    assert today[1] == 50.0
    assert today == pytest.approx((44.4, 50.0, 56.3), abs=0.1)
    assert later[2] - later[0] > today[2] - today[0]


def test_reference_change_is_asymmetric_on_the_log_scale():
    # Arrange
    analyte = "hdl"

    # Act
    fall, rise = reference_change(analyte)

    # Assert
    assert fall < 0 < rise
    assert rise > -fall


@pytest.mark.parametrize(("latest", "significant"), [(52.0, False), (65.0, True)])
def test_change_flag(latest, significant):
    # Arrange
    previous = measurement("hdl", 50.0)
    current = measurement("hdl", latest, DAY + timedelta(days=120))

    # Act
    flag = change_flag(previous, current)

    # Assert
    assert flag.significant is significant
    assert flag.percent_change == pytest.approx(100 * (latest / 50.0 - 1))
    assert flag.previous_lab_id == previous.record_id


def test_anchoring_labs_resets_on_treatment_and_pregnancy():
    # Arrange
    labs = [
        measurement("hdl", 40.0),
        measurement("sbp", 150.0),
        measurement("hdl", 45.0, DAY + timedelta(days=200)),
    ]
    after_statin = TreatmentChanges(statin_started_at=DAY + timedelta(days=30))
    after_bp = TreatmentChanges(bp_meds_started_at=DAY + timedelta(days=30))
    as_of = DAY + timedelta(days=100)

    # Act
    lipids = anchoring_labs("hdl", labs, as_of, after_statin)
    pressure = anchoring_labs("sbp", labs, as_of, after_bp)
    pregnant = anchoring_labs("sbp", labs, as_of, TreatmentChanges(pregnant=True))
    unaffected = anchoring_labs("sbp", labs, as_of, after_statin)

    # Assert
    assert lipids == []  # pre-statin lab dropped, later lab not observed yet
    assert pressure == []
    assert pregnant == []
    assert [lab.value for lab in unaffected] == [150.0]
