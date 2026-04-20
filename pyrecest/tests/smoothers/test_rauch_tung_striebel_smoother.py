import unittest

import numpy.testing as npt
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.smoothers import RauchTungStriebelSmoother


class RauchTungStriebelSmootherTest(unittest.TestCase):
    def test_filter_and_smooth_scalar_random_walk(self):
        smoother = RauchTungStriebelSmoother()
        filtered_states, predicted_states, smoothed_states, smoother_gains = (
            smoother.filter_and_smooth(
                initial_state=GaussianDistribution(array([0.0]), array([[1.0]])),
                measurements=array([1.0, 2.0]),
                measurement_matrices=array([[1.0]]),
                meas_noise_covariances=array([[1.0]]),
                system_matrices=array([[1.0]]),
                sys_noise_covariances=array([[1.0]]),
            )
        )

        self.assertEqual(len(filtered_states), 2)
        self.assertEqual(len(predicted_states), 1)
        self.assertEqual(len(smoothed_states), 2)
        self.assertEqual(len(smoother_gains), 1)

        npt.assert_allclose(filtered_states[0].mu, array([0.5]))
        npt.assert_allclose(filtered_states[1].mu, array([1.4]))
        npt.assert_allclose(predicted_states[0].mu, array([0.5]))
        npt.assert_allclose(smoothed_states[0].mu, array([0.8]))
        npt.assert_allclose(smoothed_states[1].mu, array([1.4]))

        npt.assert_allclose(filtered_states[0].C, array([[0.5]]))
        npt.assert_allclose(filtered_states[1].C, array([[0.6]]))
        npt.assert_allclose(predicted_states[0].C, array([[1.5]]))
        npt.assert_allclose(smoothed_states[0].C, array([[0.4]]))
        npt.assert_allclose(smoothed_states[1].C, array([[0.6]]))
        npt.assert_allclose(smoother_gains[0], array([[1.0 / 3.0]]))

    def test_smoothing_does_not_change_last_filtered_state(self):
        smoother = RauchTungStriebelSmoother()
        filtered_states = [
            GaussianDistribution(array([0.5]), array([[0.5]])),
            GaussianDistribution(array([1.4]), array([[0.6]])),
        ]
        predicted_states = [GaussianDistribution(array([0.5]), array([[1.5]]))]

        smoothed_states, _ = smoother.smooth(
            filtered_states=filtered_states,
            predicted_states=predicted_states,
            system_matrices=array([[1.0]]),
        )

        npt.assert_allclose(smoothed_states[-1].mu, filtered_states[-1].mu)
        npt.assert_allclose(smoothed_states[-1].C, filtered_states[-1].C)

    def test_identity_defaults_and_tuple_initial_state(self):
        smoother = RauchTungStriebelSmoother()
        filtered_states, predicted_states, smoothed_states, _ = (
            smoother.filter_and_smooth(
                initial_state=(array([0.0]), array([[1.0]])),
                measurements=array([1.0, 2.0]),
                meas_noise_covariances=array([1.0, 1.0]),
                system_matrices=array([1.0]),
                sys_noise_covariances=array([1.0]),
                sys_inputs=array([0.0]),
            )
        )

        self.assertEqual(len(filtered_states), 2)
        self.assertEqual(len(predicted_states), 1)
        self.assertEqual(len(smoothed_states), 2)

        self.assertLessEqual(
            float(smoothed_states[0].C[0, 0]),
            float(filtered_states[0].C[0, 0]),
        )

    def test_single_measurement_sequence_returns_single_smoothed_state(self):
        smoother = RauchTungStriebelSmoother()
        filtered_states, predicted_states, smoothed_states, smoother_gains = (
            smoother.filter_and_smooth(
                initial_state=GaussianDistribution(array([0.0]), array([[1.0]])),
                measurements=array([1.0]),
                measurement_matrices=eye(1),
                meas_noise_covariances=array([[1.0]]),
            )
        )

        self.assertEqual(len(filtered_states), 1)
        self.assertEqual(len(predicted_states), 0)
        self.assertEqual(len(smoothed_states), 1)
        self.assertEqual(len(smoother_gains), 0)
        npt.assert_allclose(smoothed_states[0].mu, filtered_states[0].mu)
        npt.assert_allclose(smoothed_states[0].C, filtered_states[0].C)


if __name__ == "__main__":
    unittest.main()
