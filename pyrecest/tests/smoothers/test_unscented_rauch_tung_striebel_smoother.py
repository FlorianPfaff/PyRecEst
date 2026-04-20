import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.smoothers import UnscentedRauchTungStriebelSmoother


class UnscentedRauchTungStriebelSmootherTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_filter_and_smooth_linear_identity_1d(self):
        smoother = UnscentedRauchTungStriebelSmoother()
        (
            filtered_states,
            predicted_states,
            smoothed_states,
            smoother_gains,
        ) = smoother.filter_and_smooth(
            initial_state=GaussianDistribution(array([0.0]), array([[1.0]])),
            measurements=array([1.0, 2.0, 1.5]),
            measurement_functions=lambda x: x,
            meas_noise_covariances=array([[1.0]]),
            transition_functions=lambda x, _dt: x,
            sys_noise_covariances=array([[0.5]]),
            time_steps=1.0,
        )

        self.assertEqual(len(filtered_states), 3)
        self.assertEqual(len(predicted_states), 2)
        self.assertEqual(len(smoothed_states), 3)
        self.assertEqual(len(smoother_gains), 2)

        npt.assert_allclose(filtered_states[0].mu, array([0.5]))
        npt.assert_allclose(filtered_states[1].mu, array([1.25]))
        npt.assert_allclose(filtered_states[2].mu, array([1.375]))

        npt.assert_allclose(predicted_states[0].mu, array([0.5]))
        npt.assert_allclose(predicted_states[1].mu, array([1.25]))

        npt.assert_allclose(smoothed_states[0].mu, array([0.90625]))
        npt.assert_allclose(smoothed_states[1].mu, array([1.3125]))
        npt.assert_allclose(smoothed_states[2].mu, array([1.375]))

        npt.assert_allclose(filtered_states[0].C, array([[0.5]]))
        npt.assert_allclose(filtered_states[1].C, array([[0.5]]))
        npt.assert_allclose(filtered_states[2].C, array([[0.5]]))

        npt.assert_allclose(predicted_states[0].C, array([[1.0]]))
        npt.assert_allclose(predicted_states[1].C, array([[1.0]]))

        npt.assert_allclose(smoothed_states[0].C, array([[0.34375]]))
        npt.assert_allclose(smoothed_states[1].C, array([[0.375]]))
        npt.assert_allclose(smoothed_states[2].C, array([[0.5]]))

        npt.assert_allclose(smoother_gains[0], array([[0.5]]))
        npt.assert_allclose(smoother_gains[1], array([[0.5]]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_smooth_from_filtered_matches_filter_and_smooth(self):
        smoother = UnscentedRauchTungStriebelSmoother()

        def transition_function(x, time_step):
            return array([x[0] + time_step * (0.1 * x[0] ** 2)])

        def measurement_function(x):
            return array([x[0] ** 2])

        (
            filtered_states,
            _,
            smoothed_states_direct,
            _,
        ) = smoother.filter_and_smooth(
            initial_state=(array([0.2]), array([[0.4]])),
            measurements=array([0.04, 0.5, 1.0]),
            measurement_functions=measurement_function,
            meas_noise_covariances=array([[0.2]]),
            transition_functions=transition_function,
            sys_noise_covariances=array([[0.1]]),
            time_steps=0.5,
        )

        smoothed_states_recomputed, smoother_gains = smoother.smooth_from_filtered(
            filtered_states=filtered_states,
            transition_functions=transition_function,
            sys_noise_covariances=array([[0.1]]),
            time_steps=0.5,
        )

        self.assertEqual(len(smoothed_states_recomputed), 3)
        self.assertEqual(len(smoother_gains), 2)

        for state_direct, state_recomputed in zip(
            smoothed_states_direct,
            smoothed_states_recomputed,
        ):
            npt.assert_allclose(state_direct.mu, state_recomputed.mu)
            npt.assert_allclose(state_direct.C, state_recomputed.C)

        npt.assert_allclose(
            smoothed_states_recomputed[-1].mu,
            filtered_states[-1].mu,
        )
        npt.assert_allclose(
            smoothed_states_recomputed[-1].C,
            filtered_states[-1].C,
        )
