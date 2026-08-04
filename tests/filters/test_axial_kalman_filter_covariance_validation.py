"""Regression tests for axial Kalman covariance validation."""

import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye, to_numpy
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import AxialKalmanFilter


class TestAxialKalmanFilterCovarianceValidation(unittest.TestCase):
    """Invalid unchecked Gaussian covariances must fail at the filter boundary."""

    @staticmethod
    def _unchecked_gaussian(covariance):
        return GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]),
            array(covariance),
            check_validity=False,
        )

    @staticmethod
    def _state_snapshot(filter_):
        return (
            np.asarray(to_numpy(filter_.filter_state.mu), dtype=float).copy(),
            np.asarray(to_numpy(filter_.filter_state.C), dtype=float).copy(),
        )

    def assert_state_unchanged(self, filter_, before):
        npt.assert_array_equal(to_numpy(filter_.filter_state.mu), before[0])
        npt.assert_array_equal(to_numpy(filter_.filter_state.C), before[1])

    def test_filter_state_rejects_asymmetric_covariance(self):
        filter_ = AxialKalmanFilter()
        covariance = np.eye(4)
        covariance[0, 1] = 0.25
        invalid_state = self._unchecked_gaussian(covariance)
        before = self._state_snapshot(filter_)

        with self.assertRaisesRegex(ValueError, "covariance must be symmetric"):
            filter_.filter_state = invalid_state

        self.assert_state_unchanged(filter_, before)

    def test_prediction_rejects_indefinite_covariance_atomically(self):
        filter_ = AxialKalmanFilter()
        covariance = np.eye(4)
        covariance[-1, -1] = -0.25
        invalid_noise = self._unchecked_gaussian(covariance)
        before = self._state_snapshot(filter_)

        with self.assertRaisesRegex(ValueError, "covariance must be positive definite"):
            filter_.predict_identity(invalid_noise)

        self.assert_state_unchanged(filter_, before)

    def test_update_rejects_indefinite_covariance_atomically(self):
        filter_ = AxialKalmanFilter()
        covariance = np.eye(4)
        covariance[-1, -1] = -0.25
        invalid_noise = self._unchecked_gaussian(covariance)
        before = self._state_snapshot(filter_)

        with self.assertRaisesRegex(ValueError, "covariance must be positive definite"):
            filter_.update_identity(
                invalid_noise,
                array([1.0, 0.0, 0.0, 0.0]),
            )

        self.assert_state_unchanged(filter_, before)


if __name__ == "__main__":
    unittest.main()
