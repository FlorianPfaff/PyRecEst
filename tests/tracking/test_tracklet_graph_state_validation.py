from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking.tracklet_graph import Tracklet


@pytest.mark.parametrize(
    "invalid_state",
    [
        np.array([1.0 + 2.0j, 3.0 + 4.0j]),
        [True, False],
        ["1.0", "2.0"],
        np.array([np.datetime64("2026-01-01")]),
    ],
)
def test_tracklet_rejects_non_real_numeric_state_values(invalid_state) -> None:
    state_size = np.asarray(invalid_state).size

    with pytest.raises(
        ValueError, match="start_state must contain real-valued numeric entries"
    ):
        Tracklet(
            "bad-start",
            0.0,
            1.0,
            invalid_state,
            np.zeros(state_size),
        )

    with pytest.raises(
        ValueError, match="end_state must contain real-valued numeric entries"
    ):
        Tracklet(
            "bad-end",
            0.0,
            1.0,
            np.zeros(state_size),
            invalid_state,
        )


def test_tracklet_rejects_nested_complex_object_state() -> None:
    nested_complex = np.empty(1, dtype=object)
    nested_complex[0] = np.array(1.0 + 2.0j)

    with pytest.raises(
        ValueError, match="start_state must contain real-valued numeric entries"
    ):
        Tracklet("nested-complex", 0.0, 1.0, nested_complex, [0.0])
