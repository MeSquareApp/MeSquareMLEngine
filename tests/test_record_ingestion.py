from datetime import UTC, date, datetime

import pytest

from src.components.biomarkers import ALIAS_TO_BIOMARKER, is_plausible, normalize_name
from src.components.egfr import egfr_ckd_epi_2021
from src.components.record_ingestion import (
    RecordError,
    convert_unit,
    ingest_records,
    parse_record,
    parse_timestamp,
)
from tests.conftest import START, full_panel, lab_record


@pytest.mark.parametrize(
    ("raw", "biomarker"),
    [(" HDL-C ", "hdl"), ("TC", "total_chol"), ("Hemoglobin A1c", "hba1c"),
     ("CREAT", "creatinine"), ("resting_heart_rate", "resting_heart_rate")],
)  # fmt: skip
def test_aliases_map_to_canonical_biomarkers(raw, biomarker):
    # Arrange
    name = normalize_name(raw)

    # Act
    mapped = ALIAS_TO_BIOMARKER[name]

    # Assert
    assert mapped == biomarker


def test_is_plausible():
    # Arrange / Act / Assert
    assert is_plausible("hdl", 50.0)
    assert not is_plausible("hdl", 900.0)


@pytest.mark.parametrize(
    ("biomarker", "value", "unit", "expected"),
    [
        ("hdl", 1.3, "mmol/L", 50.27),
        ("hba1c", 42.0, "mmol/mol", 5.99),
        ("creatinine", 88.42, "µmol/L", 1.0),
        ("weight", 200.0, "lbs", 90.72),
        ("egfr", 80.0, "mL/min/1.73m²", 80.0),
    ],
)
def test_convert_unit(biomarker, value, unit, expected):
    # Arrange / Act
    converted = convert_unit(biomarker, value, unit)

    # Assert
    assert converted == pytest.approx(expected, abs=0.01)


def test_parse_timestamp_accepts_trailing_z():
    # Arrange
    raw = "2026-11-03T20:50:55.898Z"

    # Act
    stamp = parse_timestamp(raw)

    # Assert
    assert stamp == datetime(2026, 11, 3, 20, 50, 55, 898000, tzinfo=UTC)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"biomarker_abbr": "XYZ"}, "unknown biomarker_abbr"),
        ({"biomarker_abbr": "LDL"}, "not used by the engine"),
        ({"unit": "g/L"}, "is not supported"),
        ({"value": 900.0}, "is outside"),
        ({"measurement_method": "self_reported"}, "is not a measured source"),
        ({"measured_at": None}, "missing fields"),
        ({"value": "high"}, "value is not a number"),
    ],
)
def test_bad_records_are_rejected_with_a_reason(change, reason):
    # Arrange
    record = {**lab_record("HDL", 50.0, "mg/dL"), **change}

    # Act / Assert
    with pytest.raises(RecordError, match=reason):
        parse_record(record)


def test_wearable_needs_a_wearable_method():
    # Arrange
    record = lab_record("resting_heart_rate", 58.0, "bpm")

    # Act / Assert
    with pytest.raises(RecordError, match="not a wearable source"):
        parse_record(record)


def test_complete_panel_derives_non_hdl_and_egfr():
    # Arrange
    records = full_panel()

    # Act
    ingested = ingest_records(records, age=52, sex=1)

    # Assert
    by_biomarker = {lab.biomarker: lab for lab in ingested.labs}
    assert by_biomarker["non_hdl"].value == 192.0
    assert by_biomarker["non_hdl"].derived_from == ("TC-2026-01-01", "HDL-2026-01-01")
    assert by_biomarker["egfr"].value == pytest.approx(egfr_ckd_epi_2021(1.0, 52, 1))
    assert ingested.rejected == []


def test_direct_value_wins_over_derived_and_partial_panel_is_kept():
    # Arrange: a partial panel with non-HDL reported directly and no HDL for TC.
    records = [
        lab_record("TC", 230.0, "mg/dL"),
        lab_record("HDL", 40.0, "mg/dL"),
        lab_record("NON_HDL", 185.0, "mg/dL"),
        lab_record("TC", 210.0, "mg/dL", date(2026, 2, 1)),
    ]

    # Act
    ingested = ingest_records(records, age=52, sex=1)

    # Assert
    non_hdl = [lab for lab in ingested.labs if lab.biomarker == "non_hdl"]
    assert [(lab.day, lab.value) for lab in non_hdl] == [(START, 185.0)]


def test_bad_unit_record_is_rejected_and_the_rest_accepted():
    # Arrange
    records = [*full_panel(), lab_record("HDL", 50.0, "g/L", record_id="bad-unit")]

    # Act
    ingested = ingest_records(records, age=52, sex=1)

    # Assert
    assert [item.record_id for item in ingested.rejected] == ["bad-unit"]
    assert len([lab for lab in ingested.labs if lab.biomarker == "hdl"]) == 1
