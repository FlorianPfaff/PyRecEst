from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking.tracklet_graph import (
    Tracklet,
    build_tracklet_adjacency,
    constant_velocity_edge_cost,
)


def test_constant_velocity_edge_keeps_large_finite_transition() -> None:
    magnitude = 1.0e308
    left = Tracklet(
        id="left",
        start_time=0.0,
        end_time=1.0,
        start_state=np.array([magnitude, 0.0]),
        end_state=np.array([magnitude, 0.0]),
    )
    right = Tracklet(
        id="right",
        start_time=2.0,
        end_time=3.0,
        start_state=np.array([0.0, magnitude]),
        end_state=np.array([0.0, magnitude]),
    )
    edge_cost = constant_velocity_edge_cost(max_speed=np.finfo(float).max)

    with np.errstate(over="raise", invalid="raise"):
        adjacency = build_tracklet_adjacency([left, right], edge_cost)

    assert len(adjacency["left"]) == 1
    target_id, cost = adjacency["left"][0]
    assert target_id == "right"
    assert np.isfinite(cost)
    assert cost == pytest.approx(np.hypot(magnitude, magnitude))
