import unittest

import numpy.testing as npt
from pyrecest.backend import array, eye
from pyrecest.smoothers import FixedLagRandomMatrixSmoother, RandomMatrixTrackerState


class FixedLagRandomMatrixSmootherTest(unittest.TestCase):
    def test_lag_one_kinematics_matches_rts(self):
        smoother = FixedLagRandomMatrixSmoother(lag=1, extent_smoothing="none")
        filtered_states = [
            RandomMatrixTrackerState(array([0.5]), array([[0.5]]), array([[2.0]]), 2.0),
            RandomMatrixTrackerState(array([1.4]), array([[0.6]]), array([[3.0]]), 3.0),
        ]
        predicted_states = [
            RandomMatrixTrackerState(array([0.5]), array([[1.5]]), array([[2.0]]), 2.0)
        ]

        smoothed_states, smoother_gains = smoother.smooth(
            filtered_states=filtered_states,
            predicted_states=predicted_states,
            system_matrices=array([[1.0]]),
        )

        npt.assert_allclose(smoothed_states[0].kinematic_state, array([0.8]))
        npt.assert_allclose(smoothed_states[0].covariance, array([[0.4]]))
        npt.assert_allclose(smoothed_states[1].kinematic_state, array([1.4]))
        npt.assert_allclose(smoothed_states[1].covariance, array([[0.6]]))
        npt.assert_allclose(smoother_gains[0][0], array([[1.0 / 3.0]]))
        npt.assert_allclose(smoothed_states[0].extent, filtered_states[0].extent)

    def test_extent_smoothing_uses_future_information_increment(self):
        smoother = FixedLagRandomMatrixSmoother(lag=1)
        filtered_states = [
            RandomMatrixTrackerState(
                array([0.0]), array([[1.0]]), array([[2.0]]), 10.0
            ),
            RandomMatrixTrackerState(
                array([0.0]), array([[1.0]]), array([[4.0]]), 13.0
            ),
        ]
        predicted_states = [
            RandomMatrixTrackerState(array([0.0]), array([[1.0]]), array([[2.0]]), 8.0)
        ]

        smoothed_states, _ = smoother.smooth(
            filtered_states=filtered_states,
            predicted_states=predicted_states,
            system_matrices=eye(1),
        )

        expected_extent = array([[(10.0 * 2.0 + 5.0 * 4.0) / 15.0]])
        npt.assert_allclose(smoothed_states[0].extent, expected_extent)
        self.assertEqual(smoothed_states[0].alpha, 15.0)

    def test_append_and_flush_emit_fixed_lag_sequence(self):
        smoother = FixedLagRandomMatrixSmoother(lag=1, extent_smoothing="none")
        first = RandomMatrixTrackerState(
            array([0.5]), array([[0.5]]), array([[2.0]]), 2.0
        )
        second = RandomMatrixTrackerState(
            array([1.4]), array([[0.6]]), array([[3.0]]), 3.0
        )
        predicted_second = RandomMatrixTrackerState(
            array([0.5]), array([[1.5]]), array([[2.0]]), 2.0
        )

        self.assertIsNone(smoother.append(first))
        emitted = smoother.append(second, predicted_second, system_matrix=eye(1))
        self.assertIsNotNone(emitted)
        npt.assert_allclose(emitted.kinematic_state, array([0.8]))

        remaining = smoother.flush()
        self.assertEqual(len(remaining), 1)
        npt.assert_allclose(remaining[0].kinematic_state, second.kinematic_state)

    def test_lag_zero_returns_filtered_states(self):
        smoother = FixedLagRandomMatrixSmoother(lag=0)
        filtered_state = RandomMatrixTrackerState(
            array([1.0]), array([[2.0]]), array([[3.0]]), 4.0
        )

        smoothed_states, smoother_gains = smoother.smooth([filtered_state])

        self.assertEqual(len(smoothed_states), 1)
        self.assertEqual(smoother_gains, [[]])
        npt.assert_allclose(
            smoothed_states[0].kinematic_state, filtered_state.kinematic_state
        )


if __name__ == "__main__":
    unittest.main()
