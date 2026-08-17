"""Regression tests for real-valued SE(2) UKF inputs."""

import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, to_numpy
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.se2_ukf import SE2UKF


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="SE2UKF update is not supported on JAX",
)
class TestSE2UKFRealInputs(unittest.TestCase):
    @staticmethod
    def _noise_distribution():
        return GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]),
            0.1 * eye(4),
        )

    def test_update_rejects_complex_measurement_without_mutating_state(self):
        current_filter = SE2UKF()
        original_mean = to_numpy(current_filter.filter_state.mu).copy()
        original_covariance = to_numpy(current_filter.filter_state.C).copy()

        with self.assertRaisesRegex(ValueError, "real-valued"):
            current_filter.update_identity(
                self._noise_distribution(),
                array([1.0 + 0.0j, 0.0, 1.0j, 0.0]),
            )

        npt.assert_allclose(to_numpy(current_filter.filter_state.mu), original_mean)
        npt.assert_allclose(
            to_numpy(current_filter.filter_state.C), original_covariance
        )

    def test_filter_state_rejects_complex_mean_direction(self):
        current_filter = SE2UKF()
        invalid_state = GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]),
            eye(4),
        )
        invalid_state.mu = array([1.0 + 0.0j, 0.0, 1.0j, 0.0])

        with self.assertRaisesRegex(ValueError, "real-valued"):
            current_filter.filter_state = invalid_state


if __name__ == "__main__":
    unittest.main()
