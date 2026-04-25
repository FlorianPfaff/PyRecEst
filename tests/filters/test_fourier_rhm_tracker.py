import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, isfinite, linalg, pi, zeros
from pyrecest.filters import FourierRHMTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Fourier RHM tracker tests use numpy.testing assertions",
)
class TestFourierRHMTracker(unittest.TestCase):
    def test_fourier_basis_uses_constant_cosine_sine_order(self):
        tracker = FourierRHMTracker(2)

        basis = tracker.fourier_basis(array(0.0))

        npt.assert_allclose(basis, array([0.5, 1.0, 0.0, 1.0, 0.0]))

    def test_radius_and_contour_points_follow_fourier_coefficients(self):
        tracker = FourierRHMTracker(
            2,
            fourier_coefficients=array([4.0, 1.0, 0.0, 0.5, 0.0]),
            kinematic_state=array([1.0, -1.0]),
        )

        npt.assert_allclose(tracker.evaluate_radius(array(0.0)), 3.5)
        npt.assert_allclose(tracker.evaluate_radius(array(pi / 2.0)), 1.5)
        npt.assert_allclose(tracker.get_extents_on_grid(4), array([3.5, 1.5, 1.5, 1.5]))
        contour_points = tracker.get_contour_points(4)

        self.assertEqual(contour_points.shape, (4, 2))
        npt.assert_allclose(contour_points[0], array([4.5, -1.0]))
        npt.assert_allclose(contour_points[1], array([1.0, 0.5]))

    def test_update_increases_radius_toward_far_measurement(self):
        tracker = FourierRHMTracker(
            0,
            initial_radius=1.0,
            coefficient_covariance=1.0,
            kinematic_covariance=1e-4,
            scale_mean=1.0,
            scale_variance=1e-4,
        )
        prior_radius = tracker.evaluate_radius(array(0.0))
        prior_covariance = tracker.covariance.copy()

        tracker.update(
            array([2.0, 0.0]),
            meas_noise_cov=0.01 * eye(2),
            scale_mean=1.0,
            scale_variance=1e-4,
        )

        self.assertGreater(tracker.evaluate_radius(array(0.0)), prior_radius)
        self.assertLess(tracker.covariance[0, 0], prior_covariance[0, 0])
        self.assertTrue(isfinite(tracker.latest_pseudo_measurement))
        self.assertGreater(tracker.latest_innovation_covariance, 0.0)
        self.assertTrue(linalg.eigvalsh(tracker.covariance)[0] > -1e-8)

    def test_update_accepts_measurements_by_row(self):
        tracker = FourierRHMTracker(
            1,
            initial_radius=1.0,
            coefficient_covariance=0.5,
            kinematic_covariance=1e-4,
            scale_mean=1.0,
            scale_variance=1e-4,
        )
        measurements_by_row = array(
            [
                [2.0, 0.0],
                [0.0, 2.0],
            ]
        )

        tracker.update(
            measurements_by_row,
            meas_noise_cov=0.01 * eye(2),
            scale_mean=1.0,
            scale_variance=1e-4,
        )

        self.assertEqual(tracker.get_point_estimate().shape, (5,))
        self.assertTrue(isfinite(tracker.latest_pseudo_measurement))

    def test_predict_linear_updates_full_state(self):
        tracker = FourierRHMTracker(1, initial_radius=1.0)
        system_matrix = eye(tracker.state_dim)
        inputs = zeros(tracker.state_dim)
        inputs[-2:] = array([1.0, -1.0])

        tracker.predict_linear(
            system_matrix, sys_noise=0.01 * eye(tracker.state_dim), inputs=inputs
        )

        npt.assert_allclose(tracker.kinematic_state, array([1.0, -1.0]))
        npt.assert_allclose(tracker.fourier_coefficients, array([2.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
