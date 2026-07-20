"""Regression tests for coupled backend and NumPy RNG state."""

from __future__ import annotations

import copy
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from pyrecest.reproducibility import temporary_seed


class _CoupledRandomBackend:
    """Backend whose seed operation also resets NumPy's global RNG."""

    def __init__(self) -> None:
        self.state = {"seed": "original"}

    def get_state(self):
        return copy.deepcopy(self.state)

    def set_state(self, state) -> None:
        self.state = copy.deepcopy(state)

    def seed(self, seed: int) -> None:
        self.state = {"seed": seed}
        np.random.seed(seed)


def test_temporary_seed_restores_numpy_state_for_coupled_backend() -> None:
    outer_numpy_state = np.random.get_state()
    try:
        np.random.seed(1234)
        expected_first = np.random.random(4)
        expected_second = np.random.random(4)

        np.random.seed(1234)
        assert_array_equal(np.random.random(4), expected_first)

        backend = _CoupledRandomBackend()
        with patch("pyrecest.reproducibility._random_backend", return_value=backend):
            with temporary_seed(7):
                np.random.random(20)
                assert backend.state == {"seed": 7}

        assert backend.state == {"seed": "original"}
        assert_array_equal(np.random.random(4), expected_second)
    finally:
        np.random.set_state(outer_numpy_state)
