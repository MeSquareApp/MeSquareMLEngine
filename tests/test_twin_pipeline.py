from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from src.components.personalization.residual_model import ResidualModel
from src.pipeline.twin_pipeline import (
    BiomarkerPrediction,
    TwinRequest,
    TwinState,
    twin_state,
)
from tests.conftest import (
    START,
    STUB_PRIOR,
    full_panel,
    lab_record,
    make_registry,
    wearable_records,
)

LATER = START + timedelta(days=120)


def by_biomarker(state: TwinState) -> dict[str, BiomarkerPrediction]:
    return {item.biomarker: item for item in state.predictions}


def test_stage0_population_for_every_biomarker(engine, twin_profile_codes, mock_prevent):
    # Arrange
    request = TwinRequest(profile=twin_profile_codes, as_of=START)

    # Act
    state = twin_state(engine, request)

    # Assert
    hdl = by_biomarker(state)["hdl"]
    assert {item.basis for item in state.predictions} == {"population"}
    assert hdl.p50 == pytest.approx(STUB_PRIOR["hdl"][0])
    assert hdl.basis_label == "Typical for your profile"
    assert state.risk.risk_basis == "estimated"
    assert state.lab_prompt.due


def test_stage1_lab_band_is_centred_on_the_lab(
    engine, twin_profile_codes, mock_prevent
):
    # Arrange
    request = TwinRequest(profile=twin_profile_codes, records=full_panel(), as_of=START)

    # Act
    state = twin_state(engine, request)

    # Assert
    predictions = by_biomarker(state)
    hdl = predictions["hdl"]
    assert hdl.basis == "lab"
    assert hdl.p50 == 38.0
    assert hdl.source_lab_id == "HDL-2026-01-01"
    assert hdl.basis_label == "Based on your Jan 1 lab"
    assert predictions["non_hdl"].p50 == 192.0
    assert state.risk.risk_basis == "measured"
    assert state.risk.inputs_used["hdl"].source == "measured"
    assert not state.lab_prompt.due
    assert state.lab_prompt.due_on == START + timedelta(days=90)


def test_lab_band_is_narrower_than_population(engine, twin_profile_codes, mock_prevent):
    # Arrange
    before = TwinRequest(profile=twin_profile_codes, as_of=START)
    after = TwinRequest(profile=twin_profile_codes, records=full_panel(), as_of=START)

    # Act
    population = by_biomarker(twin_state(engine, before))["hdl"]
    lab = by_biomarker(twin_state(engine, after))["hdl"]

    # Assert
    assert lab.p90 - lab.p10 < (population.p90 - population.p10) / 2


def test_stage2_change_flag_from_repeat_labs(engine, twin_profile_codes, mock_prevent):
    # Arrange
    records = [
        lab_record("HDL", 40.0, "mg/dL"),
        lab_record("HDL", 55.0, "mg/dL", LATER),
    ]
    request = TwinRequest(profile=twin_profile_codes, records=records, as_of=LATER)

    # Act
    hdl = by_biomarker(twin_state(engine, request))["hdl"]

    # Assert
    assert hdl.p50 == 55.0
    assert hdl.change is not None
    assert hdl.change.significant
    assert hdl.change.previous_lab_id == "HDL-2026-01-01"


def test_statin_start_resets_lipids_to_population(
    engine, twin_profile_codes, mock_prevent
):
    # Arrange
    profile = {
        **twin_profile_codes,
        "chol_meds": 1,
        "statin_started_at": START + timedelta(days=10),
    }
    request = TwinRequest(profile=profile, records=full_panel(), as_of=LATER)

    # Act
    predictions = by_biomarker(twin_state(engine, request))

    # Assert
    assert predictions["hdl"].basis == "population"
    assert predictions["non_hdl"].basis == "population"
    assert predictions["hba1c"].basis == "lab"


def test_pregnancy_returns_population_and_no_risk(
    engine, twin_profile_codes, mock_prevent
):
    # Arrange
    profile = {**twin_profile_codes, "pregnant": True}
    request = TwinRequest(profile=profile, records=full_panel(), as_of=START)

    # Act
    state = twin_state(engine, request)

    # Assert
    assert {item.basis for item in state.predictions} == {"population"}
    assert state.risk.status == "not_generated"
    assert "pregnancy" in (state.risk.reason or "")


def test_inactive_lab_stage_serves_population_but_risk_uses_the_lab(
    engine, twin_profile_codes, mock_prevent
):
    # Arrange
    never_worse = replace(engine, registry=make_registry(lab=False))
    request = TwinRequest(profile=twin_profile_codes, records=full_panel(), as_of=START)

    # Act
    state = twin_state(never_worse, request)

    # Assert
    assert by_biomarker(state)["hdl"].basis == "population"
    assert state.risk.inputs_used["hdl"].source == "measured"
    assert state.risk.inputs_used["hdl"].value == 38.0


def test_missing_height_lists_missing_inputs(engine, twin_profile_codes, mock_prevent):
    # Arrange
    profile = {**twin_profile_codes, "height_cm": None}

    # Act
    state = twin_state(engine, TwinRequest(profile=profile, as_of=START))

    # Assert
    assert state.risk.status == "not_generated"
    assert state.risk.missing_inputs == ["height_cm"]
    assert len(state.predictions) == 5


def test_rejected_records_are_reported(engine, twin_profile_codes, mock_prevent):
    # Arrange
    records = [*full_panel(), lab_record("HDL", 50.0, "g/L", record_id="bad")]

    # Act
    state = twin_state(engine, TwinRequest(profile=twin_profile_codes, records=records))

    # Assert
    assert [item.record_id for item in state.rejected] == ["bad"]


class ShiftModel(ResidualModel):
    """Residual model stub that adds 10% to the anchor."""

    def predict_band(self, features, anchor):
        return np.column_stack([anchor * 1.0, anchor * 1.1, anchor * 1.2])


def wearable_engine(engine, active: bool):
    model = ShiftModel("hba1c", "change", {}, "residual-test")
    return replace(
        engine,
        registry=make_registry(wearable=active),
        residual_models={("hba1c", "change"): model},
    )


def test_wearable_adjustment_stays_in_shadow_until_activated(
    engine, twin_profile_codes, mock_prevent
):
    # Arrange: a lab, then 70 days of resting heart rate after it.
    records = [
        lab_record("HBA1C", 6.0, "%"),
        *wearable_records(
            "resting_heart_rate", [60.0] * 70, "bpm", START + timedelta(days=1)
        ),
    ]
    request = TwinRequest(profile=twin_profile_codes, records=records)

    # Act
    shadow_state = twin_state(wearable_engine(engine, active=False), request)
    served_state = twin_state(wearable_engine(engine, active=True), request)

    # Assert
    shadow_hba1c = by_biomarker(shadow_state)["hba1c"]
    assert shadow_hba1c.basis == "lab"
    assert [item.biomarker for item in shadow_state.shadow] == ["hba1c"]
    assert shadow_state.shadow[0].shadow[1] == pytest.approx(shadow_hba1c.p50 * 1.1)
    served_hba1c = by_biomarker(served_state)["hba1c"]
    assert served_hba1c.basis == "lab_wearable"
    assert "wearable" in served_hba1c.basis_label
    assert shadow_state.model_versions["residual_hba1c_change"] == "residual-test"


def test_change_mode_needs_60_days_after_the_lab(
    engine, twin_profile_codes, mock_prevent
):
    # Arrange
    records = [
        lab_record("HBA1C", 6.0, "%"),
        *wearable_records(
            "resting_heart_rate", [60.0] * 30, "bpm", START + timedelta(days=1)
        ),
    ]

    # Act
    state = twin_state(
        wearable_engine(engine, active=False),
        TwinRequest(profile=twin_profile_codes, records=records),
    )

    # Assert
    assert state.shadow == []


def test_same_inputs_give_identical_output_and_hash(
    engine, twin_profile_codes, mock_prevent
):
    # Arrange
    records = full_panel()
    forward = TwinRequest(profile=twin_profile_codes, records=records, as_of=START)
    reordered = TwinRequest(
        profile=twin_profile_codes, records=records[::-1], as_of=START
    )

    # Act
    first, second, shuffled = (
        twin_state(engine, r) for r in (forward, forward, reordered)
    )

    # Assert
    assert first == second
    assert first.input_hash == shuffled.input_hash
    assert first.predictions == shuffled.predictions
    assert first.input_hash.startswith("sha256:")
    changed = twin_state(
        engine, TwinRequest(profile=twin_profile_codes, records=records, as_of=LATER)
    )
    assert changed.input_hash != first.input_hash


def test_bands_do_not_jump_without_new_data(engine, twin_profile_codes, mock_prevent):
    # Arrange: same lab, two consecutive days.
    day_one = TwinRequest(profile=twin_profile_codes, records=full_panel(), as_of=LATER)
    day_two = TwinRequest(
        profile=twin_profile_codes,
        records=full_panel(),
        as_of=LATER + timedelta(days=1),
    )

    # Act
    first, second = (
        by_biomarker(twin_state(engine, r))["hdl"] for r in (day_one, day_two)
    )

    # Assert
    assert second.p50 == first.p50
    assert 0 <= (second.p90 - first.p90) < 0.05


def test_profile_age_outside_prevent_range(engine, twin_profile_codes, mock_prevent):
    # Arrange
    profile = {**twin_profile_codes, "age": 25}

    # Act
    state = twin_state(engine, TwinRequest(profile=profile, as_of=START))

    # Assert
    assert state.risk.status == "not_generated"
    assert "30-79" in (state.risk.reason or "")
