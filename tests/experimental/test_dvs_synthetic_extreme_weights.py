"""Regression tests for extreme DVS synthetic-model weights."""

import numpy as np
import pytest

from pyrecest.experimental.dvs import edge_probabilities_from_activity


_EDGES = ("left", "right", "top", "bottom")


def test_edge_probabilities_scale_before_summing_extreme_weights():
    max_float = np.finfo(float).max

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        probabilities = edge_probabilities_from_activity(
            list(_EDGES),
            np.zeros(len(_EDGES), dtype=float),
            background_activity=max_float,
        )

    assert sum(probabilities.values()) == pytest.approx(1.0)
    for edge in _EDGES:
        assert probabilities[edge] == pytest.approx(0.25)
