"""Regression tests for zero-norm SE(2) sigma-point rotations."""

import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye, to_numpy
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.se2_ukf import SE2UKF


_IDENTITY = array([1.0, 0.0, 0.0, 0.0])


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="Not supported on JAX backend",
)
class TestSE2UKFZeroNormSigmaPoints(unittest.TestCase):
    def assert_finite_normalized_state(self, filter_):
        state = filter_.filter_state
        mean = np.asarray(to_numpy(state.mu), dtype=float)
        covariance = np.asarray(to_numpy(state.C), dtype=float)

        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.isfinite(covariance).all())
        npt.assert_allclose(np.linalg.norm(mean[0:2]), 1.0, atol=1e-10)

    def test_prediction_handles_zero_norm_state_and_noise_sigma_points(self):
        filter_ = SE2UKF()
        process = GaussianDistribution(_IDENTITY, eye(4) * 0.25)

        filter_.predict_identity(process)

        self.assert_finite_normalized_state(filter_)

    def test_update_handles_zero_norm_state_and_noise_sigma_points(self):
        filter_ = SE2UKF()
        filter_.filter_state = GaussianDistribution(_IDENTITY, eye(4) * 0.125)
        measurement_noise = GaussianDistribution(_IDENTITY, eye(4) * 0.125)

        filter_.update_identity(measurement_noise, _IDENTITY)

        self.assert_finite_normalized_state(filter_)


if __name__ == "__main__":
    unittest.main()
