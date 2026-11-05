import pytest

from src.components.risk import (
    InputEstimate,
    RiskProfile,
    assess_risk,
    risk_basis,
    risk_category,
)

PROFILE = RiskProfile(age=52, sex=1, smoking=0.0, bp_treated=0.0, statin=0.0)


def inputs(source: str) -> dict[str, InputEstimate]:
    band = {
        "non_hdl": (110, 170),
        "hdl": (40, 60),
        "sbp": (105, 140),
        "egfr": (75, 105),
    }
    values = {"non_hdl": 140.0, "hdl": 50.0, "sbp": 125.0, "egfr": 90.0}
    estimates = {
        name: InputEstimate(values[name], "measured")
        if source == "measured"
        else InputEstimate(values[name], "predicted", band[name])
        for name in values
    }
    estimates["hba1c"] = InputEstimate(5.5, "predicted", (5.2, 5.9))
    estimates["bmi"] = InputEstimate(27.0, "reported")
    estimates["diabetes"] = InputEstimate(0.0, "reported")
    return estimates


@pytest.mark.parametrize(
    ("risk", "category"),
    [(4.9, "low"), (5.0, "borderline"), (10.0, "intermediate"), (20.0, "high")],
)
def test_risk_category(risk, category):
    # Arrange / Act / Assert
    assert risk_category(risk) == category


def test_risk_basis():
    # Arrange
    mixed = inputs("predicted")
    mixed["hdl"] = InputEstimate(50.0, "measured")

    # Act / Assert
    assert risk_basis(inputs("measured")) == "measured"
    assert risk_basis(inputs("predicted")) == "estimated"
    assert risk_basis(mixed) == "mixed"


def test_measured_inputs_give_a_point_risk(mock_prevent):
    # Arrange
    estimates = inputs("measured")

    # Act
    assessment = assess_risk(PROFILE, estimates)

    # Assert
    ten_year = next(
        r
        for r in assessment.results
        if (r.outcome, r.horizon_years) == ("total_cvd", 10)
    )
    assert assessment.status == "generated"
    assert assessment.risk_basis == "measured"
    assert ten_year.risk_low == ten_year.risk_pct == ten_year.risk_high
    assert ten_year.category is not None
    assert {(r.outcome, r.horizon_years) for r in assessment.results} == {
        (o, h) for o in ("total_cvd", "ascvd", "heart_failure") for h in (10, 30)
    }


def test_predicted_inputs_give_a_range_around_the_point(mock_prevent):
    # Arrange
    estimates = inputs("predicted")

    # Act
    assessment = assess_risk(PROFILE, estimates)

    # Assert
    ten_year = assessment.results[0]
    assert ten_year.risk_low < ten_year.risk_pct < ten_year.risk_high
    assert assessment.risk_basis == "estimated"


def test_out_of_range_inputs_are_listed(mock_prevent):
    # Arrange
    estimates = inputs("measured")
    estimates["non_hdl"] = InputEstimate(330.0, "measured")
    estimates["bmi"] = InputEstimate(45.0, "reported")

    # Act
    assessment = assess_risk(PROFILE, estimates)

    # Assert
    assert set(assessment.out_of_range) == {"total_chol", "bmi"}
