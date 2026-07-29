import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.filters.hyperspherical_ukf import HypersphericalUKF


class HypersphericalUKFExtremeNoiseWeightsTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Arbitrary-noise prediction is not supported on this backend",
    )
    def test_extreme_finite_weights_match_scaled_equivalent(self):
        reference_filter = HypersphericalUKF(dim=2, alpha=1.0)
        extreme_filter = HypersphericalUKF(dim=2, alpha=1.0)
        noise_samples = np.array([[0.0, 1.0]])

        def direction_from_noise(_x, noise):
            if float(np.asarray(noise, dtype=float)[0]) < 0.5:
                return array([1.0, 0.0])
            return array([0.0, 1.0])

        reference_filter.predict_nonlinear_arbitrary_noise(
            direction_from_noise,
            noise_samples,
            np.array([1.0, 1.0]),
        )
        extreme_filter.predict_nonlinear_arbitrary_noise(
            direction_from_noise,
            noise_samples,
            np.array([1.0e308, 1.0e308]),
        )

        reference_mean = np.asarray(reference_filter.filter_state.mu, dtype=float)
        extreme_mean = np.asarray(extreme_filter.filter_state.mu, dtype=float)
        reference_covariance = np.asarray(reference_filter.filter_state.C, dtype=float)
        extreme_covariance = np.asarray(extreme_filter.filter_state.C, dtype=float)

        self.assertTrue(np.all(np.isfinite(extreme_mean)))
        self.assertTrue(np.all(np.isfinite(extreme_covariance)))
        npt.assert_allclose(extreme_mean, reference_mean, atol=1.0e-12)
        npt.assert_allclose(extreme_covariance, reference_covariance, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
