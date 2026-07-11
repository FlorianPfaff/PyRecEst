import unittest

import numpy as np
import numpy.testing as npt
from pyrecest import backend
from pyrecest.distributions.cart_prod.gauss_von_mises_distribution import (
    GaussVonMisesDistribution,
)
from scipy.special import ive
from scipy.stats import multivariate_normal


@unittest.skipIf(
    backend.__backend_name__ == "jax",
    reason="Gauss-von-Mises deterministic sampling is not supported on JAX",
)
class GaussVonMisesLargeKappaTest(unittest.TestCase):
    @staticmethod
    def _distribution(kappa=1000.0):
        return GaussVonMisesDistribution(2.0, 1.3, 0.4, 0.0, 0.001, kappa)

    def test_mode_density_remains_finite_for_large_kappa(self):
        kappa = 1000.0
        dist = self._distribution(kappa)

        actual = float(dist.pdf(dist.mode()))
        expected_gaussian = multivariate_normal.pdf(
            np.array([2.0]), mean=np.array([2.0]), cov=np.array([[1.3]])
        )
        expected = expected_gaussian / (2.0 * np.pi * ive(0, kappa))

        self.assertTrue(np.isfinite(actual))
        npt.assert_allclose(actual, expected, rtol=1e-12)

    def test_horwood_sigma_points_remain_finite_for_large_kappa(self):
        dist = self._distribution()

        points, weights = dist.sample_deterministic_horwood()
        points = np.asarray(backend.to_numpy(points))
        weights = np.asarray(backend.to_numpy(weights))

        self.assertTrue(np.all(np.isfinite(points)))
        self.assertTrue(np.all(np.isfinite(weights)))
        npt.assert_allclose(np.sum(weights), 1.0, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
