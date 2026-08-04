import numpy as np
import pytest
from pyrecest.experimental.dvs import simulate_rectangle_event_counts


def test_rectangle_event_simulation_handles_extreme_finite_background():
    max_float = np.finfo(np.float64).max

    with np.errstate(over="raise", invalid="raise"):
        simulation = simulate_rectangle_event_counts(
            np.array([0.0, 0.0]),
            total_events=40,
            samples_per_edge=2,
            background_activity=max_float,
            seed=4,
        )

    assert sum(simulation.observed_counts.values()) == 40
    assert simulation.true_probabilities == pytest.approx(
        {"left": 0.25, "right": 0.25, "top": 0.25, "bottom": 0.25}
    )
