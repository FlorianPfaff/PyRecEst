import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import all, array, diag, eye, isfinite, linalg
from pyrecest.filters import GGIWTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="GGIW tracker tests currently use numpy.testing assertions",
)
class TestGGIWTracker(unittest.TestCase):
    def setUp(self):
        self.kinematic_state = array([0.0, 0.0, 1.0, -1.0])
        self.covariance = diag(array([1.0, 1.0, 0.25, 0.25]))
        self.extent = diag(array([4.0, 1.0]))
        self.measurement_matrix = array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.tracker = GGIWTracker(
            self.kinematic_state,
            self.covariance,
            self.extent,
            extent_degrees_of_freedom=12.0,
            gamma_shape=4.0,
            gamma_rate=2.0,
            measurement_matrix=self.measurement_matrix,
        )

    def test_initialization_uses_extent_point_estimate(self):
        npt.assert_allclose(self.tracker.kinematic_state, self.kinematic_state)
        npt.assert_allclose(self.tracker.covariance, self.covariance)
        npt.assert_allclose(self.tracker.get_point_estimate_extent(), self.extent)
        npt.assert_allclose(self.tracker.extent_scale, 6.0 * self.extent)
        self.assertEqual(self.tracker.get_measurement_rate_estimate(), 2.0)
        self.assertEqual(self.tracker.get_point_estimate().shape[0], 9)

    def test_predict_linear_preserves_extent_mean_with_forgetting(self):
        system_matrix = array(
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.5],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        sys_noise = 0.01 * eye(4)

        self.tracker.predict_linear(
            system_matrix,
            sys_noise,
            extent_forgetting_factor=0.5,
            measurement_rate_forgetting_factor=0.5,
        )

        npt.assert_allclose(self.tracker.kinematic_state, array([0.5, -0.5, 1.0, -1.0]))
        npt.assert_allclose(self.tracker.get_point_estimate_extent(), self.extent)
        self.assertAlmostEqual(self.tracker.extent_degrees_of_freedom, 9.0)
        self.assertAlmostEqual(self.tracker.gamma_shape, 2.0)
        self.assertAlmostEqual(self.tracker.gamma_rate, 1.0)
        self.assertAlmostEqual(self.tracker.get_measurement_rate_estimate(), 2.0)

    def test_update_moves_centroid_and_updates_count_model(self):
        measurements = array(
            [
                [1.1, 0.9, 1.0, 1.0],
                [0.6, -0.4, 0.5, -0.5],
            ]
        )
        prior_covariance = self.tracker.covariance.copy()

        self.tracker.update(
            measurements,
            meas_noise_cov=0.05 * eye(2),
        )

        self.assertGreater(self.tracker.kinematic_state[0], 0.0)
        self.assertAlmostEqual(self.tracker.gamma_shape, 8.0)
        self.assertAlmostEqual(self.tracker.gamma_rate, 3.0)
        self.assertGreater(self.tracker.get_measurement_rate_estimate(), 2.0)
        self.assertAlmostEqual(self.tracker.extent_degrees_of_freedom, 16.0)
        self.assertTrue(
            all(linalg.eigvalsh(self.tracker.get_point_estimate_extent()) > 0.0)
        )
        self.assertLess(self.tracker.covariance[0, 0], prior_covariance[0, 0])
        self.assertTrue(isfinite(self.tracker.latest_log_likelihood))

    def test_update_accepts_transposed_measurement_layout(self):
        measurements_by_row = array(
            [
                [1.1, 0.6],
                [0.9, -0.4],
                [1.0, 0.5],
                [1.0, -0.5],
            ]
        )

        self.tracker.update(
            measurements_by_row,
            meas_noise_cov=0.05 * eye(2),
        )

        self.assertGreater(self.tracker.kinematic_state[0], 0.0)
        self.assertAlmostEqual(self.tracker.gamma_shape, 8.0)

    def test_get_contour_points(self):
        contour_points = self.tracker.get_contour_points(32)

        self.assertEqual(contour_points.shape, (32, 2))
        npt.assert_allclose(contour_points[0], array([2.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
