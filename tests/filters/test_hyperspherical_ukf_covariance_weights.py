import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.hyperspherical_ukf import HypersphericalUKF
from pyrecest.sampling.sigma_points import MerweScaledSigmaPoints


class HypersphericalUKFCovarianceWeightsTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Arbitrary-noise hyperspherical prediction is not supported on this backend",
    )
    def test_arbitrary_noise_prediction_uses_covariance_weights(self):
        alpha = 0.5
        beta = 2.0
        kappa = 0.0
        mean = np.array([1.0, 0.0])
        covariance = np.diag([0.04, 0.09])

        filter_ = HypersphericalUKF(
            dim=2,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )
        filter_.filter_state = GaussianDistribution(array(mean), array(covariance))
        filter_.predict_nonlinear_arbitrary_noise(
            lambda state, _noise: state,
            array([[0.0]]),
            array([1.0]),
        )

        points = MerweScaledSigmaPoints(
            n=2,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )
        sigma_points = np.asarray(points.sigma_points(mean, covariance), dtype=float)
        transformed = sigma_points / np.linalg.norm(
            sigma_points,
            axis=1,
            keepdims=True,
        )
        mean_weights = np.asarray(points.Wm, dtype=float)
        covariance_weights = np.asarray(points.Wc, dtype=float)
        transformed_mean = mean_weights @ transformed
        residuals = transformed - transformed_mean
        expected_covariance = np.einsum(
            "i,ij,ik->jk",
            covariance_weights,
            residuals,
            residuals,
        )

        actual_covariance = np.asarray(filter_.filter_state.C, dtype=float)
        npt.assert_allclose(actual_covariance, expected_covariance, atol=1e-12)
        self.assertGreater(actual_covariance[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
