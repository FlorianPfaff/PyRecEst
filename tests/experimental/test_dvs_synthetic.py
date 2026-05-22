import numpy as np
import pytest

from pyrecest.experimental.dvs import (
    count_negative_log_likelihood,
    simulate_rectangle_event_counts,
    uniform_edge_probabilities,
)


def test_motion_gated_counts_concentrate_on_vertical_edges():
    simulation = simulate_rectangle_event_counts(
        np.array([1.0, 0.0]),
        total_events=200,
        background_activity=0.0,
        seed=1,
    )

    assert simulation.observed_counts["left"] + simulation.observed_counts["right"] == 200
    assert simulation.observed_counts["top"] + simulation.observed_counts["bottom"] == 0


def test_normal_flow_likelihood_beats_uniform_for_horizontal_motion():
    simulation = simulate_rectangle_event_counts(
        np.array([1.0, 0.0]),
        total_events=200,
        background_activity=1e-3,
        seed=2,
    )

    normal_flow_nll = count_negative_log_likelihood(simulation.observed_counts, simulation.normal_flow_probabilities)
    uniform_nll = count_negative_log_likelihood(simulation.observed_counts, simulation.uniform_probabilities)

    assert normal_flow_nll < uniform_nll


def test_uniform_edge_probabilities_sum_to_one():
    probabilities = uniform_edge_probabilities(["left", "left", "right", "right", "top", "bottom"])

    assert sum(probabilities.values()) == pytest.approx(1.0)
