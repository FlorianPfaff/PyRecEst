import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,duplicate-code
import pyrecest.backend
from pyrecest.backend import abs, all, array, eye, linalg, zeros
from pyrecest.filters import (
    DecorrelatedSCGPTracker,
    DecorrelatedScGpTracker,
    FullSCGPTracker,
    GPRHMTracker,
    SCGPTracker,
    ScGpTracker,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="SCGP tracker tests currently use numpy.testing assertions",
)
class TestSCGPTracker(unittest.TestCase):
    def setUp(self):
        self.n_base_points = 8
        self.shape_state = array([1.0] * self.n_base_points)
        self.shape_covariance = 0.05 * eye(self.n_base_points)
        self.kinematic_state = array([0.0, 0.0, 0.0, 1.0, 0.1])
        self.kinematic_covariance = 1e-4 * eye(5)
        self.measurement_noise = 0.02 * eye(2)

    def _make_tracker(self, tracker_cls=FullSCGPTracker, shape_state=None):
        if shape_state is None:
            shape_state = self.shape_state
        return tracker_cls(
            self.n_base_points,
            kinematic_state=self.kinematic_state,
            kinematic_covariance=self.kinematic_covariance,
            shape_state=shape_state,
            shape_covariance=self.shape_covariance,
            measurement_noise=self.measurement_noise,
            radial_noise_variance=0.01,
            extent_forgetting_rate=0.2,
            reference_extent=self.shape_state,
        )

    def test_exports_and_aliases(self):
        self.assertIs(SCGPTracker, FullSCGPTracker)
        self.assertIs(ScGpTracker, FullSCGPTracker)
        self.assertIs(DecorrelatedScGpTracker, DecorrelatedSCGPTracker)

    def test_gprhm_contour_uses_requested_resolution(self):
        tracker = GPRHMTracker(
            n_base_points=8,
            log_prior_estimates=False,
            log_posterior_estimates=False,
            log_prior_extents=False,
            log_posterior_extents=False,
        )

        self.assertEqual(tracker.get_contour_points(7).shape, (7, 2))

    def test_predict_updates_kinematics_and_shape_process(self):
        tracker = self._make_tracker(shape_state=2.0 * self.shape_state)

        tracker.predict(dt=2.0)

        self.assertGreater(tracker.kinematic_state[0], 1.9)
        npt.assert_allclose(tracker.kinematic_state[2], 0.2, atol=1e-3)
        self.assertLess(tracker.shape_state[0], 2.0)
        npt.assert_array_less(-1e-10, linalg.eigvalsh(tracker.covariance))

    def test_full_update_creates_kinematic_shape_cross_covariance(self):
        tracker = self._make_tracker()

        tracker.update(array([1.4, 0.2]))

        cross_covariance = tracker.covariance[:5, 5:]
        self.assertFalse(all(abs(cross_covariance) <= 1e-12))
        self.assertIsNotNone(tracker.last_quadratic_form)
        npt.assert_array_less(-1e-10, linalg.eigvalsh(tracker.covariance))

    def test_decorrelated_update_keeps_cross_covariance_zero(self):
        tracker = self._make_tracker(DecorrelatedSCGPTracker)

        tracker.update(array([[1.4, 0.2], [0.1, 1.3]]))

        cross_covariance = tracker.covariance[:5, 5:]
        npt.assert_allclose(cross_covariance, zeros(cross_covariance.shape), atol=1e-12)
        npt.assert_array_less(-1e-10, linalg.eigvalsh(tracker.covariance))

    def test_active_measurement_mask_matches_single_measurement_update(self):
        masked_tracker = self._make_tracker()
        single_tracker = self._make_tracker()
        measurements = array([[1.4, 0.2], [0.1, 1.3]])

        masked_tracker.update(
            measurements,
            active_measurement_mask=array([True, False]),
        )
        single_tracker.update(measurements[0])

        npt.assert_allclose(masked_tracker.state, single_tracker.state, atol=1e-12)
        npt.assert_allclose(
            masked_tracker.covariance,
            single_tracker.covariance,
            atol=1e-12,
        )
        self.assertEqual(masked_tracker.last_active_measurement_indices, [0])
        npt.assert_allclose(masked_tracker.last_measurement_weights, array([1.0, 1.0]))

    def test_zero_measurement_weight_skips_measurement(self):
        tracker = self._make_tracker()
        state_before = array(tracker.state)
        covariance_before = array(tracker.covariance)

        tracker.update(array([1.4, 0.2]), measurement_weights=array([0.0]))

        npt.assert_allclose(tracker.state, state_before, atol=1e-12)
        npt.assert_allclose(tracker.covariance, covariance_before, atol=1e-12)
        self.assertEqual(tracker.last_active_measurement_indices, [])
        self.assertIsNone(tracker.last_quadratic_form)

    def test_measurement_weight_changes_update_strength(self):
        high_weight_tracker = self._make_tracker()
        low_weight_tracker = self._make_tracker()
        state_before = array(high_weight_tracker.state)
        measurement = array([1.4, 0.2])

        high_weight_tracker.update(measurement, measurement_weights=1.0)
        low_weight_tracker.update(measurement, measurement_weights=0.05)

        high_weight_delta = linalg.norm(high_weight_tracker.state - state_before)
        low_weight_delta = linalg.norm(low_weight_tracker.state - state_before)
        self.assertGreater(float(high_weight_delta), float(low_weight_delta))
        npt.assert_allclose(low_weight_tracker.last_measurement_weights, array([0.05]))

    def test_measurement_weights_validate_shape_and_values(self):
        tracker = self._make_tracker()

        with self.assertRaises(ValueError):
            tracker.update(
                array([[1.4, 0.2], [0.1, 1.3]]),
                measurement_weights=array([1.0]),
            )
        with self.assertRaises(ValueError):
            tracker.update(array([1.4, 0.2]), measurement_weights=-1.0)

    def test_full_tracker_contour_and_bounding_box(self):
        tracker = self._make_tracker()

        contour_points = tracker.get_contour_points(9)
        bounding_box = tracker.get_bounding_box(32)

        self.assertEqual(contour_points.shape, (9, 2))
        self.assertEqual(bounding_box["center_xy"].shape, (2,))
        self.assertEqual(bounding_box["dimension"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
