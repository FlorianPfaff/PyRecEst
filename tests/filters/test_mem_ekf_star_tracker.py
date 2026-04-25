import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,protected-access,duplicate-code
import pyrecest.backend
from pyrecest.backend import all, array, diag, eye, linalg
from pyrecest.filters import MEMEKFStarTracker, MemEkfStarTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="MEM-EKF* tracker tests currently use numpy.testing assertions",
)
class TestMEMEKFStarTracker(unittest.TestCase):
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
        self.tracker = MEMEKFStarTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
        )

    def test_initialization_and_alias(self):
        self.assertIs(MemEkfStarTracker, MEMEKFStarTracker)
        npt.assert_allclose(self.tracker.kinematic_state, self.kinematic_state)
        npt.assert_allclose(self.tracker.shape_state, self.shape_state)
        npt.assert_allclose(
            self.tracker.get_point_estimate_extent(),
            diag(array([4.0, 1.0])),
        )

    def test_shape_noise_covariance_accounts_for_shape_uncertainty(self):
        shape_noise_covariance = self.tracker._shape_noise_covariance(0.25 * eye(2))

        npt.assert_allclose(
            shape_noise_covariance,
            diag(array([0.0275, 0.06])),
            atol=1e-12,
        )

    def test_pseudo_jacobian_uses_mem_ekf_star_ordering(self):
        pseudo_jacobian = self.tracker._shape_pseudo_jacobian_star(0.25 * eye(2))

        npt.assert_allclose(
            pseudo_jacobian,
            array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.5],
                    [0.75, 0.0, 0.0],
                ]
            ),
            atol=1e-12,
        )

    def test_pseudo_measurement_covariance_is_centered(self):
        pseudo_covariance = self.tracker._pseudo_measurement_covariance(
            array([[2.0, 0.5], [0.5, 3.0]])
        )

        npt.assert_allclose(
            pseudo_covariance,
            array(
                [
                    [8.0, 0.5, 2.0],
                    [0.5, 18.0, 3.0],
                    [2.0, 3.0, 6.25],
                ]
            ),
        )

    def test_update_moves_centroid_and_updates_shape(self):
        tracker = MEMEKFStarTracker(
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
        self.assertTrue(all(linalg.eigvalsh(tracker.shape_covariance) > -1e-12))

    def test_update_accepts_multiple_measurement_layouts(self):
        tracker = MEMEKFStarTracker(
            array([0.0, 0.0, 0.0, 0.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([0.0, 1.0, 1.0]),
            diag(array([0.001, 0.5, 0.01])),
            measurement_matrix=self.measurement_matrix,
        )

        tracker.update(array([[2.0, 0.0]]), meas_noise_cov=0.01 * eye(2))

        self.assertGreater(tracker.kinematic_state[0], 0.0)
        self.assertGreater(tracker.shape_state[1], 1.0)


if __name__ == "__main__":
    unittest.main()
