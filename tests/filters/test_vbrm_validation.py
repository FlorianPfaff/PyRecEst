import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, diag, eye
from pyrecest.filters.vbrm_tracker import VBRMTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="VBRM validation tests currently use numpy.testing assertions",
)
class TestVBRMValidation(unittest.TestCase):
    def setUp(self):
        self.kinematic_state = array([0.0, 0.0, 1.0, -1.0])
        self.covariance = diag(array([0.1, 0.1, 0.01, 0.01]))
        self.shape_state = array([0.0, 2.0, 1.0])
        self.measurement_matrix = array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.measurement_noise_cov = 0.01 * eye(2)

    def _make_tracker(self, **overrides):
        arguments = {
            "kinematic_state": self.kinematic_state,
            "covariance": self.covariance,
            "shape_state": self.shape_state,
            "orientation_variance": 0.1,
            "inverse_gamma_shape": 10.0,
            "measurement_noise_cov": self.measurement_noise_cov,
            "measurement_matrix": self.measurement_matrix,
        }
        arguments.update(overrides)
        return VBRMTracker(**arguments)

    @staticmethod
    def _snapshot(tracker):
        return {
            "kinematic_state": np.asarray(tracker.kinematic_state).copy(),
            "covariance": np.asarray(tracker.covariance).copy(),
            "orientation": float(tracker.orientation),
            "orientation_variance": float(tracker.orientation_variance),
            "alpha": np.asarray(tracker.alpha).copy(),
            "beta": np.asarray(tracker.beta).copy(),
        }

    def _assert_snapshot_equal(self, tracker, snapshot):
        npt.assert_allclose(tracker.kinematic_state, snapshot["kinematic_state"])
        npt.assert_allclose(tracker.covariance, snapshot["covariance"])
        self.assertEqual(float(tracker.orientation), snapshot["orientation"])
        self.assertEqual(
            float(tracker.orientation_variance), snapshot["orientation_variance"]
        )
        npt.assert_allclose(tracker.alpha, snapshot["alpha"])
        npt.assert_allclose(tracker.beta, snapshot["beta"])

    def test_constructor_rejects_nonfinite_hyperparameters(self):
        cases = (
            ("orientation_variance", np.nan),
            ("orientation_variance", np.inf),
            ("inverse_gamma_shape", np.nan),
            ("inverse_gamma_shape", np.inf),
            ("forgetting_factor", np.nan),
            ("forgetting_factor", np.inf),
            ("extent_scale", np.nan),
            ("extent_scale", np.inf),
            ("covariance_regularization", np.nan),
            ("covariance_regularization", np.inf),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self._make_tracker(**{name: value})

    def test_constructor_rejects_nonfinite_shape_state(self):
        invalid_shapes = (
            array([np.nan, 2.0, 1.0]),
            array([np.inf, 2.0, 1.0]),
            array([0.0, np.nan, 1.0]),
            array([0.0, 2.0, np.inf]),
        )
        for shape_state in invalid_shapes:
            with self.subTest(shape_state=shape_state), self.assertRaises(ValueError):
                self._make_tracker(shape_state=shape_state)

    def test_predict_validation_failures_are_atomic(self):
        invalid_controls = (
            {"orientation_system_matrix": np.nan},
            {"orientation_sys_noise": np.inf},
            {"forgetting_factor": np.nan},
            {"forgetting_factor": 0.05},
        )
        for controls in invalid_controls:
            with self.subTest(controls=controls):
                tracker = self._make_tracker()
                snapshot = self._snapshot(tracker)
                with self.assertRaises(ValueError):
                    tracker.predict_linear(
                        2.0 * eye(4),
                        sys_noise=0.01 * eye(4),
                        **controls,
                    )
                self._assert_snapshot_equal(tracker, snapshot)

    def test_update_rejects_noninteger_iteration_overrides(self):
        for num_iterations in (True, 1.5, "2", array([2])):
            with self.subTest(num_iterations=num_iterations):
                tracker = self._make_tracker()
                snapshot = self._snapshot(tracker)
                with self.assertRaises(ValueError):
                    tracker.update(
                        array([[0.1, 0.0]]),
                        num_iterations=num_iterations,
                    )
                self._assert_snapshot_equal(tracker, snapshot)


if __name__ == "__main__":
    unittest.main()
