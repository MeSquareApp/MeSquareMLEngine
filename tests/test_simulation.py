import numpy as np
import pytest

from src.components.personalization.simulation import (
    SimulationInput,
    simulate_lab_stage,
)


def test_lab_stage_simulation_on_a_wide_population():
    # Arrange: population bands far wider than within-person variation.
    rng = np.random.default_rng(0)
    truth = np.exp(rng.normal(np.log(50), 0.25, 4000))
    population = np.tile([36.0, 50.0, 69.0], (4000, 1))

    # Act
    metrics = simulate_lab_stage(
        SimulationInput("hdl", truth, population, np.ones(4000))
    )

    # Assert
    assert metrics["error_reduction"] > 0.4
    assert 0.75 <= metrics["lab_coverage_80"] <= 0.85
    assert metrics["false_alert_rate"] == pytest.approx(0.05, abs=0.015)
    assert metrics["beats_population"]
    assert metrics["width_ratio"] >= metrics["min_width_ratio_possible"] * 0.9
