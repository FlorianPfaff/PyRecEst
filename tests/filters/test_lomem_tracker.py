import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,duplicate-code
import pyrecest.backend
from pyrecest.backend import all, array, diag, eye, linalg, pi
from pyrecest.filters import LOMEMTracker, LomemTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="LOMEM tracker tests currently use numpy.testing assertions",
)
class TestLOMEMTracker(unittest.TestCase):
    def setUp(self):
        self.state = array([0.0, 0.0, 1.0, 0.0, 2.0, 1.0])
        self.covariance = diag(array([0.5, 0.5, 0.1, 0.1, 0.2, 0.2]))
        self.measurement_noise_cov = 0.01 * eye(2)
        self.tracker = LOMEMTracker(
            self.state,
            self.covariance,
            measurement_noise_cov=self.measurement_noise_cov,
        )

    def test_initialization_and_alias(self):
        self.assertIs(LomemTracker, LOMEMTracker)
        npt.assert_allclose(self.tracker.get_point_estimate(), self.state)
        npt.assert_allclose(
            self.tracker.get_point_estimate_kinematics(), self.state[:4]
        )
        npt.assert_allclose(self.tracker.get_point_estimate_shape(), self.state[3:])
        npt.assert_allclose(
            self.tracker.get_point_estimate_shape(full_axes=True),
            array([0.0, 4.0, 2.0]),
        )
        npt.assert_allclose(self.tracker.lambda_direction, array([1.0, 0.0]))
        npt.assert_allclose(self.tracker.omicron_direction, array([0.0, 1.0]))

    def test_extent_respects_heading(self):
        tracker = LOMEMTracker(
            array([0.0, 0.0, 1.0, 0.5 * pi, 2.0, 1.0]),
            self.covariance,
        )

        npt.assert_allclose(
            tracker.get_point_estimate_extent(),
            diag(array([1.0, 4.0])),
            atol=1e-12,
        )

    def test_predict_unicycle_moves_along_lambda_direction(self):
        self.tracker.predict_unicycle(
            time_delta=0.5,
            sys_noise=0.01 * eye(6),
            longitudinal_acceleration=1.0,
            turn_rate=0.2,
        )

        npt.assert_allclose(self.tracker.state[0], 0.5)
        npt.assert_allclose(self.tracker.state[1], 0.0)
        npt.assert_allclose(self.tracker.state[2], 1.5)
        npt.assert_allclose(self.tracker.state[3], 0.1)
        self.assertTrue(all(linalg.eigvalsh(self.tracker.covariance) > 0.0))

    def test_reduce_measurements_returns_augmented_point_object_measurement(self):
        measurements = array(
            [
                [4.0, 0.0],
                [0.0, 0.0],
                [2.0, 1.0],
                [2.0, -1.0],
                [2.0, 0.0],
            ]
        )

        reduced, covariance = self.tracker.reduce_measurements(measurements)

        npt.assert_allclose(reduced[:2], array([2.0, 0.0]))
        npt.assert_allclose(reduced[2:], array([2.0, 0.5, 0.0]))
        self.assertEqual(covariance.shape, (5, 5))
        self.assertTrue(all(linalg.eigvalsh(covariance) >= -1e-12))

    def test_update_with_scan_moves_state_and_shape(self):
        tracker = LOMEMTracker(
            array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
            diag(array([5.0, 5.0, 0.1, 0.5, 1.0, 1.0])),
            measurement_noise_cov=self.measurement_noise_cov,
        )
        measurements = array(
            [
                [4.0, 0.0],
                [0.0, 0.0],
                [2.0, 1.0],
                [2.0, -1.0],
                [2.0, 0.0],
            ]
        )

        tracker.update(measurements)

        self.assertGreater(tracker.state[0], 0.0)
        self.assertGreater(tracker.state[4], 1.0)
        self.assertLess(tracker.state[5], 1.0)
        self.assertTrue(all(linalg.eigvalsh(tracker.covariance) > 0.0))

    def test_update_does_not_interpret_sensor_noise_as_extent(self):
        tracker = LOMEMTracker(
            array([0.0, 0.0, 0.0, 0.0, 2.5, 1.0]),
            diag(array([0.1, 0.1, 0.1, 0.1, 1.0, 1.0])),
            measurement_noise_cov=eye(2),
        )
        equivalent_covariance = tracker._equivalent_measurement_covariance(
            eye(2)
        )  # pylint: disable=protected-access
        x_spread = (2.0 * float(equivalent_covariance[0, 0])) ** 0.5
        y_spread = (2.0 * float(equivalent_covariance[1, 1])) ** 0.5
        measurements = array(
            [
                [x_spread, 0.0],
                [-x_spread, 0.0],
                [0.0, y_spread],
                [0.0, -y_spread],
                [0.0, 0.0],
            ]
        )
        prior_shape = tracker.state[3:].copy()

        tracker.update(measurements)

        npt.assert_allclose(tracker.state[3:], prior_shape, atol=1e-12)
        self.assertTrue(all(linalg.eigvalsh(tracker.covariance) > 0.0))

    def test_single_measurement_updates_position_only_with_diagonal_prior(self):
        tracker = LOMEMTracker(
            array([0.0, 0.0, 0.0, 0.4, 2.0, 1.0]),
            diag(array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1])),
            measurement_noise_cov=self.measurement_noise_cov,
        )

        tracker.update(array([1.0, 0.0]))

        self.assertGreater(tracker.state[0], 0.0)
        npt.assert_allclose(tracker.state[3:], array([0.4, 2.0, 1.0]))

    def test_position_speed_cross_covariance_transfers_reduced_update(self):
        covariance = diag(array([2.0, 2.0, 1.0, 0.1, 0.1, 0.1]))
        covariance[0, 2] = 0.5
        covariance[2, 0] = 0.5
        tracker = LOMEMTracker(
            array([0.0, 0.0, 0.0, 0.0, 2.0, 1.0]),
            covariance,
            measurement_noise_cov=self.measurement_noise_cov,
        )

        tracker.update(array([2.0, 0.0]))

        self.assertGreater(tracker.state[2], 0.0)

    def test_update_without_measurements_is_noop(self):
        prior_state = self.tracker.state.copy()

        self.tracker.update(array([[], []]))

        npt.assert_allclose(self.tracker.state, prior_state)

    def test_get_contour_points(self):
        contour_points = self.tracker.get_contour_points(4)

        self.assertEqual(contour_points.shape, (4, 2))
        npt.assert_allclose(contour_points[0], array([2.0, 0.0]))
        npt.assert_allclose(contour_points[1], array([0.0, 1.0]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
