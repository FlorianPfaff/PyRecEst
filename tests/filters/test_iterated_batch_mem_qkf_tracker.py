import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,protected-access,duplicate-code
import pyrecest.backend
from pyrecest.backend import all, array, diag, eye, linalg
from pyrecest.filters import (
    IteratedBatchMEMQKFTracker,
    IteratedBatchMemQkfTracker,
    MEMQKFTracker,
)


def _rot(angle):
    return array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Iterated batch MEM-QKF tests currently use numpy.testing assertions",
)
class TestIteratedBatchMEMQKFTracker(unittest.TestCase):
    def setUp(self):
        self.kinematic_state = array([0.0, 0.0, 1.0, -1.0])
        self.covariance = diag(array([0.1, 0.1, 0.01, 0.01]))
        self.shape_state = array([0.2, 2.0, 1.0])
        self.shape_covariance = diag(array([0.02, 0.1, 0.2]))
        self.measurement_matrix = array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.measurement_noise = _rot(0.35) @ diag(array([0.4, 0.9])) @ _rot(0.35).T
        self.measurements = array(
            [
                [1.8, -0.2],
                [2.4, 0.7],
                [0.4, -1.0],
                [1.0, 0.4],
            ]
        )

    def make_tracker(self, cls=IteratedBatchMEMQKFTracker, **kwargs):
        return cls(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            measurement_matrix=self.measurement_matrix,
            default_meas_noise_cov=self.measurement_noise,
            **kwargs,
        )

    def test_initialization_and_alias(self):
        tracker = self.make_tracker(update_mode="batch", n_iterations=2)

        self.assertIs(IteratedBatchMemQkfTracker, IteratedBatchMEMQKFTracker)
        self.assertEqual(tracker.update_mode, "sequential")
        self.assertEqual(tracker.n_iterations, 2)
        npt.assert_allclose(tracker.kinematic_state, self.kinematic_state)
        npt.assert_allclose(tracker.shape_state, self.shape_state)

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            self.make_tracker(n_iterations=0)
        with self.assertRaises(ValueError):
            self.make_tracker(damping=0.0)
        with self.assertRaises(ValueError):
            self.make_tracker(damping=1.1)
        with self.assertRaises(ValueError):
            self.make_tracker(minimum_batch_covariance_eigenvalue=-1.0)

    def test_single_measurement_fallback_matches_mem_qkf(self):
        measurement = array([2.0, -0.5])
        sequential_tracker = self.make_tracker(cls=MEMQKFTracker)
        iterated_batch_tracker = self.make_tracker()

        sequential_tracker.update(measurement)
        iterated_batch_tracker.update(measurement)

        npt.assert_allclose(
            iterated_batch_tracker.kinematic_state,
            sequential_tracker.kinematic_state,
        )
        npt.assert_allclose(iterated_batch_tracker.covariance, sequential_tracker.covariance)
        npt.assert_allclose(
            iterated_batch_tracker.shape_state,
            sequential_tracker.shape_state,
        )
        npt.assert_allclose(
            iterated_batch_tracker.shape_covariance,
            sequential_tracker.shape_covariance,
        )

    def test_multi_measurement_update_is_finite_and_psd(self):
        tracker = self.make_tracker(n_iterations=3)

        tracker.update(self.measurements)

        point_estimate = tracker.get_point_estimate()
        self.assertEqual(point_estimate.shape[0], 7)
        self.assertTrue(np.all(np.isfinite(point_estimate)))
        self.assertTrue(all(linalg.eigvalsh(tracker.covariance) > 0.0))
        self.assertTrue(all(linalg.eigvalsh(tracker.shape_covariance) > -1e-12))
        self.assertGreaterEqual(tracker.shape_state[1], tracker.minimum_axis_length)
        self.assertGreaterEqual(tracker.shape_state[2], tracker.minimum_axis_length)

    def test_iterated_batch_extent_update_differs_from_existing_batch_mode(self):
        existing_batch_tracker = self.make_tracker(cls=MEMQKFTracker, update_mode="batch")
        iterated_batch_tracker = self.make_tracker(n_iterations=3)

        existing_batch_tracker.update(self.measurements)
        iterated_batch_tracker.update(self.measurements)

        self.assertEqual(existing_batch_tracker.update_mode, "batch")
        self.assertEqual(iterated_batch_tracker.update_mode, "sequential")
        self.assertFalse(
            np.allclose(
                iterated_batch_tracker.shape_state,
                existing_batch_tracker.shape_state,
            )
        )
        self.assertTrue(all(linalg.eigvalsh(iterated_batch_tracker.shape_covariance) > -1e-12))

    def test_update_without_measurements_is_noop(self):
        tracker = self.make_tracker()
        prior_kinematic_state = tracker.kinematic_state.copy()
        prior_shape_state = tracker.shape_state.copy()

        tracker.update(array([[], []]))

        npt.assert_allclose(tracker.kinematic_state, prior_kinematic_state)
        npt.assert_allclose(tracker.shape_state, prior_shape_state)


if __name__ == "__main__":
    unittest.main()
