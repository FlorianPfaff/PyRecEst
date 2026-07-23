# pylint: disable=no-name-in-module,no-member
import unittest

import mpmath
import numpy as np
import pyrecest.backend
from pyrecest.backend import array, complex128, to_numpy
from pyrecest.distributions import ComplexWatsonDistribution


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="ComplexWatsonDistribution.log_norm is not supported on JAX",
)
class TestComplexWatsonNegativeKappa(unittest.TestCase):
    @staticmethod
    def _reference_log_norm(dim, kappa):
        with mpmath.workdps(80):
            log_reciprocal_norm = (
                mpmath.log(2.0)
                + dim * mpmath.log(mpmath.pi)
                - mpmath.loggamma(dim)
                + mpmath.log(mpmath.hyp1f1(1, dim, mpmath.mpf(kappa)))
            )
        return float(-log_reciprocal_norm)

    def test_negative_log_norm_matches_hypergeometric_reference(self):
        cases = (
            (1, -100.0),
            (2, -10.0),
            (3, -100.0),
            (10, -20.0),
        )

        for dim, kappa in cases:
            with self.subTest(dim=dim, kappa=kappa):
                actual = ComplexWatsonDistribution.log_norm(dim, kappa)
                expected = self._reference_log_norm(dim, kappa)
                self.assertAlmostEqual(actual, expected, places=11)

    def test_negative_log_norm_array_is_finite_and_accurate(self):
        dim = 3
        kappas = array([-1000.0, -100.0, -20.0, -5.0, -1.0, -0.1])
        actual = np.asarray(to_numpy(ComplexWatsonDistribution.log_norm(dim, kappas)))
        expected = np.asarray(
            [self._reference_log_norm(dim, float(kappa)) for kappa in to_numpy(kappas)]
        )

        self.assertTrue(np.all(np.isfinite(actual)))
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

    def test_negative_concentration_pdf_remains_finite(self):
        distribution = ComplexWatsonDistribution(
            array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex128), -100.0
        )
        points = array(
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
            dtype=complex128,
        )
        densities = np.asarray(to_numpy(distribution.pdf(points)))

        self.assertTrue(np.all(np.isfinite(densities)))
        self.assertTrue(np.all(densities >= 0.0))
        self.assertGreater(densities[1], densities[0])
