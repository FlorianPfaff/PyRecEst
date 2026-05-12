import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,protected-access,duplicate-code
import pyrecest.backend
from pyrecest.backend import all, array, diag, eye, linalg
from pyrecest.filters import MEMQKFTracker, MemQkfTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="MEM-QKF tracker tests currently use numpy.testing assertions",
)
class TestMEMQKFTracker(unittest.TestCase):
    def setUp(self):
        self.kinematic_state = array([0.0, 0.0, 1.0, -1.0])
        self.covariance = diag(array([0.1, 0.1, 0.01, 0.01]))
        self.shape_state = array([0.0, 2.0, 1.0])
        self.shape_covariance = diag(array([0.01, 0.1, 0.2]))
        self.measurement_matrix = array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.tracker = MEMQKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
        )

    def test_initialization_and_alias(self):
        self.assertIs(MemQkfTracker, MEMQKFTracker)
        npt.assert_allclose(self.tracker.kinematic_state, self.kinematic_state)
        npt.assert_allclose(self.tracker.shape_state, self.shape_state)
        npt.assert_allclose(
            self.tracker.get_point_estimate_extent(),
            diag(array([4.0, 1.0])),
        )
        self.assertEqual(self.tracker.get_point_estimate().shape[0], 7)

    def test_initialization_drops_orientation_axis_cross_covariances(self):
        shape_covariance = array(
            [
                [0.2, 0.05, -0.04],
                [0.05, 0.4, 0.01],
                [-0.04, 0.01, 0.3],
            ]
        )
        tracker = MEMQKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            shape_covariance,
            measurement_matrix=self.measurement_matrix,
        )

        npt.assert_allclose(tracker.shape_covariance[0, 1:], array([0.0, 0.0]))
        npt.assert_allclose(tracker.shape_covariance[1:, 0], array([0.0, 0.0]))
        npt.assert_allclose(tracker.shape_covariance[1:, 1:], shape_covariance[1:, 1:])

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

    def test_update_moves_centroid_and_updates_axes(self):
        tracker = MEMQKFTracker(
            array([0.0, 0.0, 0.0, 0.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([0.0, 1.0, 1.0]),
            diag(array([0.001, 0.5, 0.01])),
            measurement_matrix=self.measurement_matrix,
        )
        prior_axis_covariance = tracker.shape_covariance[1, 1]

        tracker.update(array([2.0, 0.0]), meas_noise_cov=0.01 * eye(2))

        self.assertGreater(tracker.kinematic_state[0], 0.0)
        self.assertGreater(tracker.shape_state[1], 1.0)
        self.assertLess(tracker.shape_covariance[1, 1], prior_axis_covariance)
        self.assertTrue(all(linalg.eigvalsh(tracker.covariance) > 0.0))
        self.assertTrue(all(linalg.eigvalsh(tracker.shape_covariance) > -1e-12))

    def test_update_accepts_multiple_measurement_layouts(self):
        tracker = MEMQKFTracker(
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


if __name__ == "__main__":
    unittest.main()
