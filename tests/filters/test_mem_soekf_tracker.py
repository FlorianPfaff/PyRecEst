import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,protected-access,duplicate-code
import pyrecest.backend
from pyrecest.backend import all, array, diag, eye, linalg
from pyrecest.filters import MEMSOEKFTracker, MemSoekfTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="MEM-SOEKF tracker tests currently use numpy.testing assertions",
)
class TestMEMSOEKFTracker(unittest.TestCase):
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
        self.tracker = MEMSOEKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
        )

    def test_initialization_and_alias(self):
        self.assertIs(MemSoekfTracker, MEMSOEKFTracker)
        npt.assert_allclose(self.tracker.kinematic_state, self.kinematic_state)
        npt.assert_allclose(self.tracker.shape_state, self.shape_state)
        npt.assert_allclose(
            self.tracker.get_point_estimate_extent(),
            diag(array([4.0, 1.0])),
        )

    def test_pseudo_jacobian_uses_soekf_ordering(self):
        pseudo_jacobian = self.tracker._shape_pseudo_jacobian_soekf(0.25 * eye(2))

        npt.assert_allclose(
            pseudo_jacobian,
            array(
                [
                    [0.0, 1.0, 0.0],
                    [0.75, 0.0, 0.0],
                    [0.0, 0.0, 0.5],
                ]
            ),
            atol=1e-12,
        )

    def test_pseudo_hessians_include_shape_moment_curvature(self):
        pseudo_hessians = self.tracker._shape_pseudo_hessians_soekf(0.25 * eye(2))

        self.assertEqual(pseudo_hessians.shape, (3, 3, 3))
        npt.assert_allclose(pseudo_hessians[0, 0, 0], -1.5, atol=1e-5)
        npt.assert_allclose(pseudo_hessians[0, 1, 1], 0.5, atol=1e-5)
        npt.assert_allclose(pseudo_hessians[2, 0, 0], 1.5, atol=1e-5)
        npt.assert_allclose(pseudo_hessians[2, 2, 2], 0.5, atol=1e-5)

    def test_update_moves_centroid_and_updates_shape(self):
        tracker = MEMSOEKFTracker(
            array([0.0, 0.0, 0.0, 0.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([0.0, 1.0, 1.0]),
            diag(array([0.001, 0.5, 0.01])),
            measurement_matrix=self.measurement_matrix,
            covariance_regularization=1e-9,
        )
        prior_shape_covariance = tracker.shape_covariance.copy()

        tracker.update(array([2.0, 0.0]), meas_noise_cov=0.01 * eye(2))

        self.assertGreater(tracker.kinematic_state[0], 0.0)
        self.assertGreater(tracker.shape_state[1], 1.0)
        self.assertLess(tracker.shape_covariance[1, 1], prior_shape_covariance[1, 1])
        self.assertTrue(all(linalg.eigvalsh(tracker.covariance) > -1e-10))
        self.assertTrue(all(linalg.eigvalsh(tracker.shape_covariance) > -1e-10))

    def test_update_accepts_multiple_measurement_layouts(self):
        tracker = MEMSOEKFTracker(
            array([0.0, 0.0, 0.0, 0.0]),
            diag(array([0.1, 0.1, 0.01, 0.01])),
            array([0.0, 1.0, 1.0]),
            diag(array([0.001, 0.5, 0.01])),
            measurement_matrix=self.measurement_matrix,
            covariance_regularization=1e-9,
        )

        tracker.update(array([[2.0, 0.0]]), meas_noise_cov=0.01 * eye(2))

        self.assertGreater(tracker.kinematic_state[0], 0.0)
        self.assertGreater(tracker.shape_state[1], 1.0)


if __name__ == "__main__":
    unittest.main()
