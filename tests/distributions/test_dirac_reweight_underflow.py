"""Regression tests for numerically stable Dirac reweighting."""

import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import LinearDiracDistribution


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Subnormal float64 regression is specific to the NumPy backend.",
)
class TestDiracReweightUnderflow(unittest.TestCase):
    def test_reweigh_preserves_the_only_positive_support_intersection(self):
        smallest_subnormal = np.nextafter(0.0, 1.0)
        distribution = LinearDiracDistribution(
            array([[0.0], [1.0], [2.0]]),
            array([smallest_subnormal, 1.0, 0.0]),
        )

        reweighted = distribution.reweigh(
            lambda _: array([smallest_subnormal, 0.0, 1.0])
        )

        npt.assert_array_equal(reweighted.w, array([1.0, 0.0, 0.0]))
        npt.assert_array_equal(
            distribution.w,
            array([smallest_subnormal, 1.0, 0.0]),
        )


if __name__ == "__main__":
    unittest.main()
