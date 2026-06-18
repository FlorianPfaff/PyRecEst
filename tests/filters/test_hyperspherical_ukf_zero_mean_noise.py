import unittest

import numpy as np

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.hyperspherical_ukf import HypersphericalUKF


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="HypersphericalUKF prediction/update are supported only on the NumPy backend.",
)
class HypersphericalUKFZeroMeanNoiseTest(unittest.TestCase):
    def setUp(self):
        self.ukf = HypersphericalUKF(dim=3)
        self.ukf.filter_state = GaussianDistribution(
            array([1.0, 0.0, 0.0]), eye(3), check_validity=False
        )

    def test_predict_rejects_nonzero_system_noise_mean(self):
        nonzero_system_noise = GaussianDistribution(
            array([0.1, 0.0, 0.0]), 0.01 * eye(3), check_validity=False
        )

        with self.assertRaisesRegex(ValueError, "gauss_sys.*zero mean"):
            self.ukf.predict_identity(nonzero_system_noise)

    def test_update_rejects_nonzero_measurement_noise_mean(self):
        nonzero_measurement_noise = GaussianDistribution(
            array([0.0, 0.1, 0.0]), 0.01 * eye(3), check_validity=False
        )

        with self.assertRaisesRegex(ValueError, "gauss_meas.*zero mean"):
            self.ukf.update_identity(nonzero_measurement_noise, array([1.0, 0.0, 0.0]))

    def test_zero_mean_gaussian_noise_still_runs(self):
        zero_mean_noise = GaussianDistribution(
            array([0.0, 0.0, 0.0]), 0.01 * eye(3), check_validity=False
        )

        self.ukf.predict_identity(zero_mean_noise)
        self.ukf.update_identity(zero_mean_noise, array([0.0, 1.0, 0.0]))

        estimate = np.asarray(self.ukf.get_point_estimate(), dtype=float)
        self.assertTrue(np.isfinite(estimate).all())


if __name__ == "__main__":
    unittest.main()
