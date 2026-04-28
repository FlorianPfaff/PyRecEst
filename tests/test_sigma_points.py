import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import __backend_name__
from pyrecest.sampling import JulierSigmaPoints, MerweScaledSigmaPoints


@unittest.skipIf(
    __backend_name__ in ("pytorch", "jax"),
    reason="Sigma-point tests use NumPy assertions",
)
class TestSigmaPoints(unittest.TestCase):
    def test_merwe_scaled_sigma_points_match_moments(self):
        mean = np.array([1.0, -2.0])
        covariance = np.array([[2.0, 0.4], [0.4, 1.0]])
        points = MerweScaledSigmaPoints(n=2, alpha=0.5, beta=2.0, kappa=0.0)

        sigmas = points.sigma_points(mean, covariance)
        weighted_mean = points.Wm @ sigmas
        deviations = sigmas - weighted_mean
        weighted_covariance = deviations.T @ (deviations * points.Wc[:, None])

        self.assertEqual(sigmas.shape, (5, 2))
        npt.assert_allclose(weighted_mean, mean)
        npt.assert_allclose(weighted_covariance, covariance)

    def test_julier_sigma_points_match_moments(self):
        mean = np.array([0.5, 1.5])
        covariance = np.array([[1.5, 0.2], [0.2, 0.5]])
        points = JulierSigmaPoints(n=2, kappa=1.0)

        sigmas = points.sigma_points(mean, covariance)
        weighted_mean = points.Wm @ sigmas
        deviations = sigmas - weighted_mean
        weighted_covariance = deviations.T @ (deviations * points.Wc[:, None])

        self.assertEqual(sigmas.shape, (5, 2))
        npt.assert_allclose(weighted_mean, mean)
        npt.assert_allclose(weighted_covariance, covariance)


if __name__ == "__main__":
    unittest.main()
