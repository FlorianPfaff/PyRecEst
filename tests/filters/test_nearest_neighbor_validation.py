import unittest
from unittest.mock import patch

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, dstack, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Nearest-neighbor validation tests require the numpy backend fixtures.",
)
class NearestNeighborValidationTest(unittest.TestCase):
    def _tracker(self, n_targets=1):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = [
            KalmanFilter(
                GaussianDistribution(
                    array([float(target_no + 1), 0.0, 0.0, 0.0]), eye(4)
                )
            )
            for target_no in range(n_targets)
        ]
        return tracker

    def _assert_filter_states_equal(self, expected_states, tracker):
        actual_states = tracker.filter_state
        self.assertEqual(len(actual_states), len(expected_states))
        for expected, actual in zip(expected_states, actual_states):
            npt.assert_array_equal(actual.mu, expected.mu)
            npt.assert_array_equal(actual.C, expected.C)

    def test_duplicate_filter_handles_raise_value_error(self):
        shared_filter = KalmanFilter(GaussianDistribution(zeros(4), eye(4)))
        tracker = GlobalNearestNeighbor()

        with self.assertRaisesRegex(ValueError, "same handle"):
            tracker.filter_state = [shared_filter, shared_filter]

    def test_predict_linear_rejects_nonzero_mean_gaussian_system_noise(self):
        tracker = self._tracker()
        nonzero_mean_noise = GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]), eye(4), check_validity=False
        )

        with self.assertRaisesRegex(ValueError, "zero mean"):
            tracker.predict_linear(eye(4), nonzero_mean_noise)

    def test_predict_linear_rejects_vector_system_matrix(self):
        tracker = self._tracker(n_targets=2)
        original_states = tracker.filter_state

        with self.assertRaisesRegex(ValueError, "system_matrices"):
            tracker.predict_linear(array([2.0, 0.0, 0.0, 0.0]), array([[0.0]]))

        self._assert_filter_states_equal(original_states, tracker)

    def test_predict_linear_rejects_short_per_target_inputs_atomically(self):
        cases = (
            (
                "system_matrices",
                dstack((2.0 * eye(4),)),
                eye(4),
                None,
            ),
            (
                "sys_noises",
                2.0 * eye(4),
                dstack((zeros((4, 4)),)),
                None,
            ),
            (
                "inputs",
                eye(4),
                zeros((4, 4)),
                array([[1.0, 1.0, 1.0, 1.0]]).T,
            ),
        )

        for error_field, system_matrices, sys_noises, inputs in cases:
            with self.subTest(error_field=error_field):
                tracker = self._tracker(n_targets=2)
                original_states = tracker.filter_state

                with self.assertRaisesRegex(ValueError, error_field):
                    tracker.predict_linear(system_matrices, sys_noises, inputs)

                self._assert_filter_states_equal(original_states, tracker)

    def test_predict_linear_rejects_mixed_track_dimensions_atomically(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = [
            KalmanFilter(
                GaussianDistribution(array([1.0, 0.0, 0.0, 0.0]), eye(4))
            ),
            KalmanFilter(GaussianDistribution(array([2.0, 0.0]), eye(2))),
        ]
        original_states = tracker.filter_state

        with self.assertRaisesRegex(ValueError, "same state dimension"):
            tracker.predict_linear(2.0 * eye(4), eye(4))

        self._assert_filter_states_equal(original_states, tracker)

    def test_update_linear_unsupported_backend_raises_not_implemented(self):
        tracker = self._tracker()

        with patch.object(pyrecest.backend, "__backend_name__", "jax"):
            with self.assertRaisesRegex(NotImplementedError, "numpy backend"):
                tracker.update_linear(array([[0.0], [0.0]]), eye(4)[:2, :], eye(2))

    def test_update_linear_dimension_mismatch_raises_value_error(self):
        tracker = self._tracker()

        with self.assertRaisesRegex(ValueError, "measurement matrix"):
            tracker.update_linear(array([[0.0]]), eye(4)[:2, :], eye(2))


if __name__ == "__main__":
    unittest.main()
