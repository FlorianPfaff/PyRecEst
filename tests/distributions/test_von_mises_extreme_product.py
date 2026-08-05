"""Regression tests for overflow-safe von Mises concentration products."""

import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.distributions import VonMisesDistribution


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Strict floating-point overflow regression uses the NumPy backend",
)
class TestVonMisesExtremeProduct(unittest.TestCase):
    def test_multiplication_by_uniform_preserves_extreme_concentration(self):
        concentrated = VonMisesDistribution(0.0, 1.0e308)
        uniform = VonMisesDistribution(1.3, 0.0)

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            product = concentrated.multiply(uniform)

        self.assertTrue(np.isfinite(float(product.kappa)))
        npt.assert_allclose(float(product.mu), 0.0, rtol=0.0, atol=0.0)
        npt.assert_allclose(float(product.kappa), 1.0e308, rtol=1.0e-15)

    def test_orthogonal_extreme_concentrations_keep_finite_resultant(self):
        first = VonMisesDistribution(0.0, 1.0e308)
        second = VonMisesDistribution(np.pi / 2.0, 1.0e308)

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            product = first.multiply(second)

        expected_kappa = np.hypot(1.0e308, 1.0e308)
        self.assertTrue(np.isfinite(float(product.kappa)))
        npt.assert_allclose(float(product.mu), np.pi / 4.0, rtol=1.0e-15)
        npt.assert_allclose(float(product.kappa), expected_kappa, rtol=1.0e-15)


if __name__ == "__main__":
    unittest.main()
