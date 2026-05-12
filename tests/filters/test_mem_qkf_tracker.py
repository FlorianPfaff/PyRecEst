import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,protected-access,duplicate-code
import pyrecest.backend
from pyrecest.backend import all, array, diag, eye, linalg
from pyrecest.filters import MEMQKFTracker, MemQkfTracker


def _rot(angle):
    return array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )


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
        self.assertEqual(self.tracker.update_mode, "sequential")
        npt.assert_allclose(self.tracker.kinematic_state, self.kinematic_state)
        npt.assert_allclose(self.tracker.shape_state, self.shape_state)
        npt.assert_allclose(
            self.tracker.get_point_estimate_extent(),
            diag(array([4.0, 1.0])),
        )
        self.assertEqual(self.tracker.get_point_estimate().shape[0], 7)

    def test_rejects_unknown_update_mode(self):
        with self.assertRaises(ValueError):
            MEMQKFTracker(
                self.kinematic_state,
                self.covariance,
                self.shape_state,
                self.shape_covariance,
                measurement_matrix=self.measurement_matrix,
                update_mode="unsupported",
            )

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

    def test_batch_update_uses_single_centroid_kinematic_update(self):
        R = _rot(0.4) @ diag(array([0.4, 1.1])) @ _rot(0.4).T
        measurements = array([[1.7, -0.4], [2.3, 0.8], [0.6, -1.1], [1.1, 0.2]])
        tracker = MEMQKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
            update_mode="batch",
        )

        normalized_measurements = tracker._normalize_measurements(measurements)
        centroid = np.mean(normalized_measurements, axis=1)
        extent_transform = tracker._extent_transform()
        centroid_covariance = (
            extent_transform @ tracker.multiplicative_noise_cov @ extent_transform.T + R
        ) / normalized_measurements.shape[1]
        innovation_covariance = (
            self.measurement_matrix @ self.covariance @ self.measurement_matrix.T
            + centroid_covariance
        )
        kinematic_gain = tracker._gain_from_cross_covariance(
            self.covariance @ self.measurement_matrix.T,
            innovation_covariance,
        )
        expected_state = self.kinematic_state + kinematic_gain @ (
            centroid - self.measurement_matrix @ self.kinematic_state
        )
        expected_covariance = self.covariance - (
            kinematic_gain @ innovation_covariance @ kinematic_gain.T
        )

        tracker.update(measurements, meas_noise_cov=R)

        npt.assert_allclose(tracker.kinematic_state, expected_state)
        npt.assert_allclose(tracker.covariance, expected_covariance)
        self.assertTrue(all(linalg.eigvalsh(tracker.shape_covariance) > -1e-12))

    def test_batch_update_differs_from_sequential_on_measurement_sets(self):
        R = _rot(0.5) @ diag(array([0.7, 0.2])) @ _rot(0.5).T
        measurements = array([[1.5, -0.3], [2.0, 0.6], [-0.2, 1.4], [0.8, -1.0]])
        sequential_tracker = MEMQKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
        )
        batch_tracker = MEMQKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
            update_mode="batch",
        )

        sequential_tracker.update(measurements, meas_noise_cov=R)
        batch_tracker.update(measurements, meas_noise_cov=R)

        self.assertFalse(
            np.allclose(
                batch_tracker.kinematic_state,
                sequential_tracker.kinematic_state,
            )
        )
        self.assertTrue(all(linalg.eigvalsh(batch_tracker.covariance) > 0.0))

    def test_update_without_measurements_is_noop(self):
        prior_kinematic_state = self.tracker.kinematic_state.copy()
        prior_shape_state = self.tracker.shape_state.copy()

        self.tracker.update(array([[], []]), meas_noise_cov=0.01 * eye(2))

        npt.assert_allclose(self.tracker.kinematic_state, prior_kinematic_state)
        npt.assert_allclose(self.tracker.shape_state, prior_shape_state)

    def test_default_measurement_noise_matches_explicit_noise(self):
        R = _rot(0.35) @ diag(array([0.5, 1.3])) @ _rot(0.35).T
        explicit_tracker = MEMQKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
        )
        default_tracker = MEMQKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
            default_meas_noise_cov=R,
        )

        explicit_tracker.update(array([2.0, -0.5]), meas_noise_cov=R)
        default_tracker.update(array([2.0, -0.5]))

        npt.assert_allclose(
            default_tracker.kinematic_state, explicit_tracker.kinematic_state
        )
        npt.assert_allclose(default_tracker.covariance, explicit_tracker.covariance)
        npt.assert_allclose(default_tracker.shape_state, explicit_tracker.shape_state)
        npt.assert_allclose(
            default_tracker.shape_covariance, explicit_tracker.shape_covariance
        )

    def test_set_r_updates_default_measurement_noise(self):
        R = _rot(0.2) @ diag(array([0.3, 0.9])) @ _rot(0.2).T
        tracker_via_setter = MEMQKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
        )
        tracker_via_constructor = MEMQKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
            default_meas_noise_cov=R,
        )

        tracker_via_setter.set_R(R)
        tracker_via_setter.update(array([1.5, -0.2]))
        tracker_via_constructor.update(array([1.5, -0.2]))

        npt.assert_allclose(
            tracker_via_setter.get_point_estimate(),
            tracker_via_constructor.get_point_estimate(),
        )
        npt.assert_allclose(
            tracker_via_setter.shape_covariance,
            tracker_via_constructor.shape_covariance,
        )

    def test_canonicalizes_negative_axes_and_axis_covariance(self):
        tracker = MEMQKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
        )

        axes, axis_covariance = tracker._canonicalize_axes_and_axis_covariance(
            array([-2.0, 3.0]),
            array([[4.0, 1.5], [1.5, 9.0]]),
        )

        npt.assert_allclose(axes, array([2.0, 3.0]))
        npt.assert_allclose(axis_covariance, array([[4.0, -1.5], [-1.5, 9.0]]))

    def test_projects_covariance_to_positive_semidefinite_cone(self):
        tracker = MEMQKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
        )

        projected = tracker._project_symmetric_covariance(
            array([[1.0, 2.0], [2.0, 1.0]])
        )

        self.assertTrue(all(linalg.eigvalsh(projected) >= -1e-12))
        npt.assert_allclose(projected, projected.T)

    def test_matches_reference_single_measurement_with_rotated_noise(self):
        R = _rot(0.35) @ diag(array([0.5, 1.3])) @ _rot(0.35).T
        tracker = MEMQKFTracker(
            array([1.0, -2.0, 0.5, -0.25]),
            diag(array([0.7, 0.5, 0.2, 0.15])),
            array([0.4, 3.0, 1.5]),
            array(
                [
                    [0.12, 0.03, -0.02],
                    [0.03, 0.4, 0.05],
                    [-0.02, 0.05, 0.3],
                ]
            ),
            measurement_matrix=self.measurement_matrix,
            default_meas_noise_cov=R,
        )

        tracker.update(array([4.2, -0.3]))

        npt.assert_allclose(
            tracker.kinematic_state,
            array([1.640726536537, -1.726294883847, 0.5, -0.25]),
            rtol=1e-11,
            atol=1e-11,
        )
        npt.assert_allclose(
            tracker.covariance,
            array(
                [
                    [0.548779436205, 0.014872785112, 0.0, 0.0],
                    [0.014872785112, 0.39950147325, 0.0, 0.0],
                    [0.0, 0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.0, 0.15],
                ]
            ),
            rtol=1e-11,
            atol=1e-11,
        )
        npt.assert_allclose(
            tracker.shape_state,
            array([0.421699822463, 3.175376001946, 1.460534411964]),
            rtol=1e-11,
            atol=1e-11,
        )
        npt.assert_allclose(
            tracker.shape_covariance,
            array(
                [
                    [0.116431846933, 0.0, 0.0],
                    [0.0, 0.388418700999, 0.050000621341],
                    [0.0, 0.050000621341, 0.296641000868],
                ]
            ),
            rtol=1e-11,
            atol=1e-11,
        )

    def test_matches_reference_multiple_measurements_with_axis_covariance(self):
        R = _rot(0.75) @ diag(array([0.8, 0.25])) @ _rot(0.75).T
        tracker = MEMQKFTracker(
            array([-1.0, 0.7, 0.4, 0.2]),
            array(
                [
                    [0.8, 0.1, 0.0, 0.02],
                    [0.1, 0.6, 0.01, 0.0],
                    [0.0, 0.01, 0.2, 0.03],
                    [0.02, 0.0, 0.03, 0.25],
                ]
            ),
            array([-0.7, 2.5, 0.9]),
            array(
                [
                    [0.18, 0.02, 0.01],
                    [0.02, 0.35, -0.04],
                    [0.01, -0.04, 0.22],
                ]
            ),
            measurement_matrix=self.measurement_matrix,
        )

        tracker.update(
            array([[0.6, -1.2], [1.4, 0.1], [-0.8, 1.6], [2.1, -0.4]]),
            meas_noise_cov=R,
        )

        npt.assert_allclose(
            tracker.kinematic_state,
            array([0.169678397452, 0.432365872634, 0.39295584805, 0.231002997924]),
            rtol=1e-11,
            atol=1e-11,
        )
        npt.assert_allclose(
            tracker.covariance,
            array(
                [
                    [0.237025136168, -0.030283754173, -0.001019776956, 0.006180572643],
                    [-0.030283754173, 0.202133177335, 0.00350499824, -0.001633343414],
                    [-0.001019776956, 0.00350499824, 0.199891616517, 0.030001601447],
                    [0.006180572643, -0.001633343414, 0.030001601447, 0.249654113954],
                ]
            ),
            rtol=1e-11,
            atol=1e-11,
        )
        npt.assert_allclose(
            tracker.shape_state,
            array([-0.710280286611, 2.503516324213, 0.796103049888]),
            rtol=1e-11,
            atol=1e-11,
        )
        npt.assert_allclose(
            tracker.shape_covariance,
            array(
                [
                    [0.109712822241, 0.0, 0.0],
                    [0.0, 0.261238659416, -0.039961799629],
                    [0.0, -0.039961799629, 0.20388307481],
                ]
            ),
            rtol=1e-11,
            atol=1e-11,
        )


if __name__ == "__main__":
    unittest.main()
