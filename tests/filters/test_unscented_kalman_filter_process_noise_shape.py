"""Regression tests for UKF process-noise covariance shape handling."""

import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="UnscentedKalmanFilter is not supported on this backend",
)
class UnscentedKalmanFilterProcessNoiseShapeTest(unittest.TestCase):
    def test_predict_rejects_vector_process_noise_without_mutating_state(self):
        initial_mean = array([0.5, -0.25])
        initial_covariance = array([[1.2, 0.3], [0.3, 0.8]])
        ukf = UnscentedKalmanFilter(
            GaussianDistribution(initial_mean, initial_covariance)
        )

        with self.assertRaisesRegex(ValueError, "process noise covariance Q"):
            ukf.predict_identity(array([0.4, 0.2]))

        npt.assert_allclose(ukf.get_point_estimate(), initial_mean)
        npt.assert_allclose(ukf.filter_state.covariance(), initial_covariance)

    def test_predict_accepts_length_one_process_noise_for_one_dimensional_state(self):
        ukf = UnscentedKalmanFilter(GaussianDistribution(array([0.5]), array([[1.25]])))

        ukf.predict_identity(array([0.5]))

        npt.assert_allclose(ukf.get_point_estimate(), array([0.5]))
        npt.assert_allclose(ukf.filter_state.covariance(), array([[1.75]]))


if __name__ == "__main__":
    unittest.main()
