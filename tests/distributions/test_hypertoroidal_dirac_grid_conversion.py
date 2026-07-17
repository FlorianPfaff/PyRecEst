from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
import pytest
from pyrecest.backend import array, zeros_like
from pyrecest.distributions import (
    AbstractHypertoroidalDistribution,
    HypertoroidalDiracDistribution,
)


class _GridBackedHypertoroidalDistribution(AbstractHypertoroidalDistribution):
    def __init__(self, grid_values):
        super().__init__(2)
        self.grid_values = array(grid_values)
        self._grid = array([[0.0, 0.0], [1.0, 1.0]])

    def pdf(self, xs):
        return zeros_like(xs)

    def get_grid(self):
        return self._grid


def test_from_distribution_rejects_negative_grid_values() -> None:
    distribution = _GridBackedHypertoroidalDistribution([-1.0, -2.0])

    with pytest.raises(ValueError, match="nonnegative"):
        HypertoroidalDiracDistribution.from_distribution(distribution)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Extreme float64 normalization regression is specific to NumPy.",
)
def test_from_distribution_stabilizes_extreme_finite_grid_values() -> None:
    maximum = np.finfo(float).max
    distribution = _GridBackedHypertoroidalDistribution([maximum, maximum / 2.0])

    converted = HypertoroidalDiracDistribution.from_distribution(distribution)

    npt.assert_allclose(
        np.asarray(converted.w),
        np.array([2.0 / 3.0, 1.0 / 3.0]),
        rtol=1.0e-15,
        atol=0.0,
    )
