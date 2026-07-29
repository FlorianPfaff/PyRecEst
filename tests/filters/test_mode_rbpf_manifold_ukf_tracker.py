import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend as pyrecest_backend
from pyrecest.filters import (
    ModeRBPFManifoldUKF,
    ModeRBPFManifoldUKFTracker,
    ModeRbpfManifoldUkfTracker,
)


@unittest.skipIf(
    pyrecest_backend.__backend_name__ != "numpy",
    reason="ModeRBPFManifoldUKFTracker is currently NumPy-backend only",
)
class TestModeRBPFManifoldUKFTracker(unittest.TestCase):
    def setUp(self):
        self.kinematic_state = np.array([0.0, 0.0, 1.0, -0.5])
        self.covariance = np.diag([0.2, 0.2, 0.05, 0.05])
        self.shape_state = np.array([0.2, 2.0, 1.0])
        self.shape_covariance = np.diag([0.05, 0.1, 0.1])
        self.meas_noise_cov = 0.05 * np.eye(2)
        self.sys_noise = 0.01 * np.eye(4)
        self.shape_sys_noise = np.diag([0.01, 0.01, 0.01])
        self.measurements = np.array(
            [
                [1.2, 0.1],
                [0.8, -0.2],
                [1.0, 0.2],
                [1.3, -0.1],
            ]
        )

    def make_tracker(self, **kwargs):
        parameters = {
            "meas_noise_cov": self.meas_noise_cov,
            "sys_noise": self.sys_noise,
            "shape_sys_noise": self.shape_sys_noise,
            "n_particles": 24,
            "rng": 0,
            "resampling_threshold": 12,
        }
        parameters.update(kwargs)
        return ModeRBPFManifoldUKFTracker(
            self.kinematic_state,
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            **parameters,
        )

    def test_aliases_are_exported(self):
        self.assertIs(ModeRbpfManifoldUkfTracker, ModeRBPFManifoldUKFTracker)
        self.assertIs(ModeRBPFManifoldUKF, ModeRBPFManifoldUKFTracker)

    def test_validates_particle_count(self):
        invalid_values = (
            True,
            np.bool_(True),
            1.5,
            np.array(1.5),
            np.array([3]),
            0,
            -1,
        )
        for invalid in invalid_values:
            with self.subTest(n_particles=invalid):
                with self.assertRaisesRegex(ValueError, "n_particles"):
                    self.make_tracker(n_particles=invalid)

        for valid in (np.int64(3), np.array(3)):
            with self.subTest(n_particles=valid):
                tracker = self.make_tracker(n_particles=valid)
                self.assertEqual(tracker.n_particles, 3)
                self.assertEqual(tracker.weights.shape, (3,))

    def test_predict_update_smoke(self):
        tracker = self.make_tracker()

        posterior = tracker.update(self.measurements)
        prior = tracker.predict()
        state = tracker.get_state()
        semi_axis_state = tracker.get_state(full_axis_lengths=False)
        covariance_state, covariance = tracker.get_state_and_cov()
        extent = tracker.get_point_estimate_extent()
        contour = tracker.get_contour_points(16)
        mode_probabilities = tracker.get_mode_probabilities()

        self.assertEqual(posterior.shape, (7,))
        self.assertEqual(prior.shape, (7,))
        self.assertEqual(state.shape, (7,))
        self.assertEqual(semi_axis_state.shape, (7,))
        self.assertEqual(covariance_state.shape, (7,))
        self.assertEqual(covariance.shape, (7, 7))
        self.assertEqual(extent.shape, (2, 2))
        self.assertEqual(contour.shape, (16, 2))
        self.assertTrue(np.all(np.isfinite(state)))
        self.assertTrue(np.all(np.isfinite(covariance)))
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) >= -1e-10))
        self.assertTrue(np.all(np.linalg.eigvalsh(extent) >= -1e-10))
        self.assertTrue(np.isclose(np.sum(tracker.weights), 1.0))
        self.assertTrue(np.isclose(sum(mode_probabilities.values()), 1.0))
        self.assertEqual(set(mode_probabilities), {"free", "velocity", "maneuver"})
        npt.assert_allclose(state[-2:], 2.0 * semi_axis_state[-2:])

    def test_static_mode_initialization_prefers_free_mode(self):
        tracker = ModeRBPFManifoldUKFTracker(
            np.array([0.0, 0.0, 0.0, 0.0]),
            self.covariance,
            self.shape_state,
            self.shape_covariance,
            meas_noise_cov=self.meas_noise_cov,
            sys_noise=self.sys_noise,
            shape_sys_noise=self.shape_sys_noise,
            n_particles=50,
            rng=1,
        )

        mode_probabilities = tracker.get_mode_probabilities()

        self.assertGreater(mode_probabilities["free"], mode_probabilities["velocity"])
        self.assertTrue(np.isclose(sum(mode_probabilities.values()), 1.0))

    def test_axes_remain_positive(self):
        tracker = self.make_tracker()

        for _ in range(3):
            tracker.update(self.measurements)
            tracker.predict()

        state = tracker.get_state(full_axis_lengths=False)
        state_array = tracker.get_state_array(with_weight=True)

        self.assertTrue(np.all(state[5:7] > 0.0))
        self.assertTrue(np.all(state_array[:, 5:7] > 0.0))
        self.assertEqual(state_array.shape[1], 8)

    def test_original_parameter_constructor(self):
        tracker = ModeRBPFManifoldUKFTracker.from_original_parameters(
            m_init=self.kinematic_state,
            p_init=self.shape_state,
            p_kinematic_init=self.covariance,
            p_shape_init=self.shape_covariance,
            r=self.meas_noise_cov,
            q_kinematic=self.sys_noise,
            q_shape=self.shape_sys_noise,
            n_particles=8,
            rng=2,
        )

        self.assertIsInstance(tracker, ModeRBPFManifoldUKFTracker)
        self.assertEqual(tracker.get_point_estimate().shape, (7,))
        self.assertEqual(tracker.get_state_array(with_weight=True).shape, (8, 8))

    def test_update_without_measurements_is_noop(self):
        tracker = self.make_tracker()
        prior = tracker.get_point_estimate().copy()

        posterior = tracker.update(np.empty((0, 2)))

        npt.assert_allclose(posterior, prior)

    def test_preserves_impossible_mode_probabilities(self):
        transition_matrix = np.eye(3)
        tracker = self.make_tracker(
            transition_matrix=transition_matrix,
            initial_mode_probs=[1.0, 0.0, 0.0],
            n_particles=8,
        )

        npt.assert_array_equal(tracker.transition_matrix, transition_matrix)
        npt.assert_array_equal(tracker.modes, np.zeros(8, dtype=int))
        npt.assert_array_equal(
            tracker._normalize_probs([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        tracker._propagate_modes()
        npt.assert_array_equal(tracker.modes, np.zeros(8, dtype=int))

    def test_rejects_invalid_transition_probabilities(self):
        invalid_matrices = (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, -0.1, 0.1],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, np.nan, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, np.inf, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.eye(2),
            np.array([1.0, 0.0, 0.0]),
        )
        for transition_matrix in invalid_matrices:
            with self.subTest(transition_matrix=transition_matrix):
                with self.assertRaisesRegex(ValueError, "transition_matrix"):
                    self.make_tracker(transition_matrix=transition_matrix)

    def test_rejects_invalid_initial_mode_probabilities(self):
        invalid_probabilities = (
            [0.0, 0.0, 0.0],
            [1.0, -0.1, 0.1],
            [1.0, np.nan, 0.0],
            [1.0, np.inf, 0.0],
            [1.0, 0.0],
        )
        for initial_mode_probs in invalid_probabilities:
            with self.subTest(initial_mode_probs=initial_mode_probs):
                with self.assertRaisesRegex(ValueError, "initial_mode_probs"):
                    self.make_tracker(initial_mode_probs=initial_mode_probs)


if __name__ == "__main__":
    unittest.main()
