import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, pi
from pyrecest.experimental.dvs import (
    DVSFullSCGPTracker,
    DVSPointProcessSCGPTracker,
    DVSSCGPTracker,
    PointProcessUpdateConfig,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="DVS SCGP tracker tests currently use numpy.testing assertions",
)
class TestDVSFullSCGPTracker(unittest.TestCase):
    def _make_tracker(self, inactive_activity_threshold=0.0):
        n_base_points = 16
        return DVSFullSCGPTracker(
            n_base_points,
            kinematic_state=array([0.0, 0.0, 0.0, 0.0, 0.0]),
            kinematic_covariance=1e-4 * eye(5),
            shape_state=array([1.0] * n_base_points),
            shape_covariance=0.05 * eye(n_base_points),
            measurement_noise=0.01 * eye(2),
            inactive_activity_threshold=inactive_activity_threshold,
            radial_noise_variance=0.0,
        )

    def test_exports_alias(self):
        self.assertIs(DVSSCGPTracker, DVSFullSCGPTracker)

    def test_horizontal_motion_activates_left_and_right_contours(self):
        tracker = self._make_tracker()

        activities = tracker.contour_event_activity(
            angles=array([0.0, pi / 2.0, pi, 3.0 * pi / 2.0]),
            event_velocity=array([1.0, 0.0]),
        )

        self.assertGreater(activities[0], 0.99)
        self.assertLess(activities[1], 1e-6)
        self.assertGreater(activities[2], 0.99)
        self.assertLess(activities[3], 1e-6)

    def test_update_skips_inactive_contour_measurements(self):
        tracker = self._make_tracker(inactive_activity_threshold=0.25)

        tracker.update(
            array(
                [
                    [1.2, 0.0],
                    [0.0, 1.2],
                ]
            ),
            event_velocity=array([1.0, 0.0]),
        )

        self.assertEqual(tracker.last_active_measurement_indices, [0])
        self.assertGreater(tracker.last_event_activities[0], 0.99)
        self.assertLess(tracker.last_event_activities[1], 1e-6)

    def test_zero_event_activity_floor_is_allowed_when_inactive_events_are_skipped(
        self,
    ):
        tracker = self._make_tracker(inactive_activity_threshold=0.25)

        tracker.update(
            array(
                [
                    [1.2, 0.0],
                    [0.0, 1.2],
                ]
            ),
            event_velocity=array([1.0, 0.0]),
            event_activity_floor=0.0,
        )

        self.assertEqual(tracker.last_active_measurement_indices, [0])

    def test_update_can_infer_event_velocity_from_kinematics(self):
        tracker = DVSFullSCGPTracker(
            8,
            kinematic_state=array([0.0, 0.0, 0.0, 2.0, 0.0]),
            kinematic_covariance=1e-4 * eye(5),
            shape_state=array([1.0] * 8),
            shape_covariance=0.05 * eye(8),
            measurement_noise=0.01 * eye(2),
        )

        activities = tracker.contour_event_activity(angles=array([0.0, pi / 2.0]))

        npt.assert_allclose(activities, array([1.0, 0.0]), atol=1e-6)

    def test_point_process_update_records_likelihood_terms(self):
        tracker = DVSPointProcessSCGPTracker(
            8,
            kinematic_state=array([0.0, 0.0, 0.0, 0.0, 0.0]),
            kinematic_covariance=1e-4 * eye(5),
            shape_state=array([1.0] * 8),
            shape_covariance=0.05 * eye(8),
            measurement_noise=0.01 * eye(2),
            radial_noise_variance=0.0,
            point_process_update_config=PointProcessUpdateConfig(
                contour_samples=16,
                finite_difference_eps=1e-3,
                map_step_size=0.01,
                max_map_iterations=1,
                shape_update_modes=2,
            ),
        )

        tracker.update(
            array(
                [
                    [1.0, -0.1],
                    [1.0, 0.1],
                ]
            ),
            event_velocity=array([1.0, 0.0]),
        )

        self.assertIsNotNone(tracker.last_event_likelihood_terms)
        self.assertEqual(tracker.last_event_likelihood_terms.event_count, 2)
        self.assertIsNotNone(tracker.last_event_likelihood_gradient)

    def test_point_process_empty_batch_resets_previous_diagnostics(self):
        tracker = DVSPointProcessSCGPTracker(
            8,
            kinematic_state=array([0.0, 0.0, 0.0, 0.0, 0.0]),
            kinematic_covariance=1e-4 * eye(5),
            shape_state=array([1.0] * 8),
            shape_covariance=0.05 * eye(8),
            measurement_noise=0.01 * eye(2),
            radial_noise_variance=0.0,
            point_process_update_config=PointProcessUpdateConfig(
                contour_samples=16,
                finite_difference_eps=1e-3,
                map_step_size=0.01,
                max_map_iterations=1,
                shape_update_modes=2,
            ),
        )

        tracker.update(
            array(
                [
                    [1.0, -0.1],
                    [1.0, 0.1],
                ]
            ),
            event_velocity=array([1.0, 0.0]),
        )

        self.assertEqual(tracker.last_active_measurement_indices, [0, 1])
        self.assertGreater(np.asarray(tracker.last_event_activities).size, 0)

        empty_event_velocity = array([0.0, 1.0])
        tracker.update(array(np.empty((0, 2))), event_velocity=empty_event_velocity)

        self.assertIsNotNone(tracker.last_event_likelihood_terms)
        self.assertEqual(tracker.last_event_likelihood_terms.event_count, 0)
        self.assertEqual(
            tracker.last_event_log_likelihood,
            tracker.last_event_likelihood_terms.log_likelihood,
        )
        self.assertEqual(tracker.last_active_measurement_indices, [])
        self.assertEqual(np.asarray(tracker.last_event_activities).shape, (0,))
        npt.assert_allclose(
            tracker.last_event_likelihood_gradient,
            np.zeros_like(np.asarray(tracker.state, dtype=float)),
            atol=0.0,
        )
        npt.assert_allclose(
            tracker.last_event_likelihood_state_update,
            np.zeros_like(np.asarray(tracker.state, dtype=float)),
            atol=0.0,
        )
        self.assertIsNone(tracker.last_quadratic_form)


if __name__ == "__main__":
    unittest.main()
