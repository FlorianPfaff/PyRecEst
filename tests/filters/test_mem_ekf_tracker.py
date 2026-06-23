import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,duplicate-code
import pyrecest.backend
from pyrecest.backend import all, array, diag, eye, linalg, pi
from pyrecest.filters import MEMEKFTracker, MemEkfTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="MEM-EKF tracker tests currently use numpy.testing assertions",
)
class TestMEMEKFTracker(unittest.TestCase):
    def setUp(self):
        self.kinematic_state = array([0.0, 0.0, 1.0, -1.0])
        self.covariance = diag(array([0.1, 0.1, 0.01, 0.01]))
        self.shape_state = array([0.0, 2.0, 1.0])
        self.shape_covariance = diag(array([0.01, 0.1, 0.1]))
        self.measurement_matrix = array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.tracker = MEMEKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
        )

    def test_initialization_and_alias(self):
        self.assertIs(MemEkfTracker, MEMEKFTracker)
        npt.assert_allclose(self.tracker.kinematic_state, self.kinematic_state)
        npt.assert_allclose(self.tracker.covariance, self.covariance)
        npt.assert_allclose(self.tracker.get_point_estimate_shape(), self.shape_state)
        npt.assert_allclose(
            self.tracker.get_point_estimate_extent(),
            diag(array([4.0, 1.0])),
        )
        self.assertEqual(self.tracker.get_point_estimate().shape[0], 7)

    def test_rejects_nonfinite_covariance_regularization(self):
        for bad_value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(covariance_regularization=bad_value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    MEMEKFTracker(
                        self.kinematic_state,
                        self.covariance,
                        self.shape_state,
                        self.shape_covariance,
                        measurement_matrix=self.measurement_matrix,
                        covariance_regularization=bad_value,
                    )

    def test_extent_respects_orientation(self):
        tracker = MEMEKFTracker(
            self.kinematic_state,
            self.covariance,
            array([0.5 * pi, 2.0, 1.0]),
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
        )

        npt.assert_allclose(
            tracker.get_point_estimate_extent(),
            diag(array([1.0, 4.0])),
            atol=1e-12,
        )

    def test_predict_linear_moves_kinematics_and_keeps_shape_by_default(self):
        system_matrix = array(
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.5],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.tracker.predict_linear(system_matrix, 0.01 * eye(4))

        npt.assert_allclose(
            self.tracker.kinematic_state,
            array([0.5, -0.5, 1.0, -1.0]),
        )
        npt.assert_allclose(self.tracker.shape_state, self.shape_state)
        npt.assert_allclose(
            self.tracker.get_point_estimate_extent(),
            diag(array([4.0, 1.0])),
        )

    def test_update_moves_centroid_and_updates_shape(self):
        tracker = MEMEKFTracker(
            array([0.0, 0.0, 0.0, 0.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([0.0, 1.0, 1.0]),
            diag(array([0.001, 0.5, 0.01])),
            measurement_matrix=self.measurement_matrix,
        )
        prior_shape_covariance = tracker.shape_covariance.copy()

        tracker.update(array([2.0, 0.0]), meas_noise_cov=0.01 * eye(2))

        self.assertGreater(tracker.kinematic_state[0], 0.0)
        self.assertGreater(tracker.shape_state[1], 1.0)
        self.assertLess(tracker.shape_covariance[1, 1], prior_shape_covariance[1, 1])
        self.assertTrue(all(linalg.eigvalsh(tracker.covariance) > 0.0))
        self.assertTrue(all(linalg.eigvalsh(tracker.shape_covariance) > 0.0))

    def test_update_accepts_transposed_measurement_layout(self):
        tracker = MEMEKFTracker(
            array([0.0, 0.0, 0.0, 0.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([0.0, 1.0, 1.0]),
            diag(array([0.001, 0.5, 0.01])),
            measurement_matrix=self.measurement_matrix,
        )

        tracker.update(array([[2.0, 0.0]]), meas_noise_cov=0.01 * eye(2))

        self.assertGreater(tracker.kinematic_state[0], 0.0)
        self.assertGreater(tracker.shape_state[1], 1.0)

    def test_update_without_measurements_is_noop(self):
        prior_kinematic_state = self.tracker.kinematic_state.copy()
        prior_shape_state = self.tracker.shape_state.copy()

        self.tracker.update(array([[], []]), meas_noise_cov=0.01 * eye(2))

        npt.assert_allclose(self.tracker.kinematic_state, prior_kinematic_state)
        npt.assert_allclose(self.tracker.shape_state, prior_shape_state)

    def test_update_without_flat_empty_measurements_is_noop(self):
        prior_kinematic_state = self.tracker.kinematic_state.copy()
        prior_shape_state = self.tracker.shape_state.copy()

        self.tracker.update(array([]), meas_noise_cov=0.01 * eye(2))

        npt.assert_allclose(self.tracker.kinematic_state, prior_kinematic_state)
        npt.assert_allclose(self.tracker.shape_state, prior_shape_state)

    def test_get_contour_points(self):
        contour_points = self.tracker.get_contour_points(4)

        self.assertEqual(contour_points.shape, (4, 2))
        npt.assert_allclose(contour_points[0], array([2.0, 0.0]))
        npt.assert_allclose(contour_points[1], array([0.0, 1.0]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
