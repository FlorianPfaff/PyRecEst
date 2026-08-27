import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, zeros
from pyrecest.filters import FourierRHMTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Fourier RHM tracker validation tests use numpy.testing assertions",
)
class TestFourierRHMTrackerValidation(unittest.TestCase):
    def test_predict_linear_validation_failure_is_atomic(self):
        tracker = FourierRHMTracker(1)
        state_before = tracker.get_point_estimate().copy()
        covariance_before = tracker.covariance.copy()
        system_matrix = 2.0 * eye(tracker.state_dim)

        with self.assertRaises(ValueError):
            tracker.predict_linear(system_matrix, sys_noise=zeros((2, 2)))

        npt.assert_allclose(tracker.get_point_estimate(), state_before)
        npt.assert_allclose(tracker.covariance, covariance_before)

    def test_process_noise_covariance_is_not_silently_symmetrized(self):
        tracker = FourierRHMTracker(1)
        covariance_before = tracker.covariance.copy()
        asymmetric_noise = eye(tracker.state_dim)
        asymmetric_noise[0, 1] = 0.5

        with self.assertRaises(ValueError):
            tracker.predict_identity(asymmetric_noise)

        npt.assert_allclose(tracker.covariance, covariance_before)

    def test_update_rejects_nonfinite_measurement_noise_atomically(self):
        tracker = FourierRHMTracker(0)
        state_before = tracker.get_point_estimate().copy()
        covariance_before = tracker.covariance.copy()
        invalid_noise = array([[0.01, 0.0], [0.0, np.nan]])

        with self.assertRaises(ValueError):
            tracker.update(array([2.0, 0.0]), meas_noise_cov=invalid_noise)

        npt.assert_allclose(tracker.get_point_estimate(), state_before)
        npt.assert_allclose(tracker.covariance, covariance_before)

    def test_constructor_rejects_nonfinite_scalar_controls(self):
        for keyword in (
            "scale_mean",
            "scale_variance",
            "ukf_alpha",
            "ukf_beta",
            "ukf_kappa",
            "covariance_regularization",
        ):
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                FourierRHMTracker(0, **{keyword: np.nan})

    def test_update_rejects_nonfinite_scale_overrides(self):
        tracker = FourierRHMTracker(0)
        state_before = tracker.get_point_estimate().copy()
        covariance_before = tracker.covariance.copy()

        for keyword in ("scale_mean", "scale_variance"):
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                tracker.update(
                    array([2.0, 0.0]),
                    meas_noise_cov=0.01 * eye(2),
                    **{keyword: np.nan},
                )
            npt.assert_allclose(tracker.get_point_estimate(), state_before)
            npt.assert_allclose(tracker.covariance, covariance_before)


if __name__ == "__main__":
    unittest.main()
