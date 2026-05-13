import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,protected-access,duplicate-code
import pyrecest.backend
from pyrecest.backend import all, arctan2, array, diag, eye, linalg
from pyrecest.filters.mem_ekf_star_oa_tracker import (
    MEMEKFStarOATracker,
    MemEkfStarOATracker,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="MEM-EKF*-OA tracker tests currently use numpy.testing assertions",
)
class TestMEMEKFStarOATracker(unittest.TestCase):
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
        self.tracker = MEMEKFStarOATracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
        )

    def test_initialization_and_alias(self):
        self.assertIs(MemEkfStarOATracker, MEMEKFStarOATracker)
        npt.assert_allclose(self.tracker.kinematic_state, self.kinematic_state)
        npt.assert_allclose(
            self.tracker.shape_state[0],
            arctan2(self.kinematic_state[3], self.kinematic_state[2]),
        )
        npt.assert_allclose(self.tracker.shape_state[1:], self.shape_state[1:])

    def test_extent_jacobians_zero_orientation_column(self):
        first_row_jacobian, second_row_jacobian = self.tracker._extent_row_jacobians()

        npt.assert_allclose(first_row_jacobian[:, 0], array([0.0, 0.0]))
        npt.assert_allclose(second_row_jacobian[:, 0], array([0.0, 0.0]))

    def test_pseudo_jacobian_uses_velocity_heading(self):
        tracker = MEMEKFStarOATracker(
            array([0.0, 0.0, 1.0, 0.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([1.5, 2.0, 1.0]),
            diag(array([0.01, 0.1, 0.2])),
            measurement_matrix=self.measurement_matrix,
        )

        pseudo_jacobian = tracker._shape_pseudo_jacobian_star(0.25 * eye(2))

        npt.assert_allclose(
            pseudo_jacobian,
            array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.5],
                    [0.0, 0.0, 0.0],
                ]
            ),
            atol=1e-12,
        )

    def test_predict_realigns_orientation_mean(self):
        tracker = MEMEKFStarOATracker(
            array([0.0, 0.0, 0.0, 1.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([0.0, 2.0, 1.0]),
            diag(array([0.01, 0.1, 0.2])),
            measurement_matrix=self.measurement_matrix,
        )
        system_matrix = array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0, 0.0],
            ]
        )

        tracker.predict_linear(system_matrix, sys_noise=0.01 * eye(4))

        npt.assert_allclose(
            tracker.shape_state[0],
            arctan2(tracker.kinematic_state[3], tracker.kinematic_state[2]),
        )

    def test_update_moves_centroid_and_keeps_heading_orientation(self):
        tracker = MEMEKFStarOATracker(
            array([0.0, 0.0, 1.0, 0.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([1.2, 1.0, 1.0]),
            diag(array([0.001, 0.5, 0.01])),
            measurement_matrix=self.measurement_matrix,
        )
        prior_shape_covariance = tracker.shape_covariance.copy()

        tracker.update(array([2.0, 0.0]), meas_noise_cov=0.01 * eye(2))

        self.assertGreater(tracker.kinematic_state[0], 0.0)
        npt.assert_allclose(
            tracker.shape_state[0],
            arctan2(tracker.kinematic_state[3], tracker.kinematic_state[2]),
        )
        self.assertLess(tracker.shape_covariance[1, 1], prior_shape_covariance[1, 1])
        self.assertTrue(all(linalg.eigvalsh(tracker.covariance) > 0.0))
        self.assertTrue(all(linalg.eigvalsh(tracker.shape_covariance) > -1e-12))


if __name__ == "__main__":
    unittest.main()
