import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, reshape
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.hyperspherical_ukf import HypersphericalUKF


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Not supported on this backend",
)
class HypersphericalUKFVectorShapeValidationTest(unittest.TestCase):
    @staticmethod
    def _zero_mean_noise(dim):
        return GaussianDistribution(
            array(np.zeros(dim)),
            array(0.1 * np.eye(dim)),
        )

    @staticmethod
    def _assert_initial_state(filter_instance):
        npt.assert_allclose(filter_instance.filter_state.mu, array([1.0, 0.0]))
        npt.assert_allclose(filter_instance.filter_state.C, array(np.eye(2)))

    def test_predict_rejects_matrix_transition_outputs_without_state_change(self):
        for output_shape in ((1, 2), (2, 1)):
            with self.subTest(output_shape=output_shape):
                filter_instance = HypersphericalUKF(dim=2)

                with self.assertRaisesRegex(
                    ValueError,
                    "transition function output must be scalar or one-dimensional",
                ):
                    filter_instance.predict_nonlinear(
                        lambda x, shape=output_shape: reshape(x, shape),
                        self._zero_mean_noise(2),
                    )

                self._assert_initial_state(filter_instance)

    def test_update_rejects_matrix_measurement_outputs_without_state_change(self):
        for output_shape in ((1, 2), (2, 1)):
            with self.subTest(output_shape=output_shape):
                filter_instance = HypersphericalUKF(dim=2)

                with self.assertRaisesRegex(
                    ValueError,
                    "measurement function output must be scalar or one-dimensional",
                ):
                    filter_instance.update_nonlinear(
                        lambda x, shape=output_shape: reshape(x, shape),
                        self._zero_mean_noise(2),
                        array([1.0, 0.0]),
                    )

                self._assert_initial_state(filter_instance)

    def test_update_rejects_matrix_measurements_without_state_change(self):
        for measurement_shape in ((1, 2), (2, 1)):
            with self.subTest(measurement_shape=measurement_shape):
                filter_instance = HypersphericalUKF(dim=2)
                measurement = reshape(array([1.0, 0.0]), measurement_shape)

                with self.assertRaisesRegex(
                    ValueError,
                    "measurement z must be scalar or one-dimensional",
                ):
                    filter_instance.update_identity(
                        self._zero_mean_noise(2),
                        measurement,
                    )

                self._assert_initial_state(filter_instance)

    def test_arbitrary_noise_rejects_matrix_weight_vectors(self):
        for weights in (
            array([[1.0, 1.0]]),
            array([[1.0], [1.0]]),
        ):
            with self.subTest(weight_shape=weights.shape):
                filter_instance = HypersphericalUKF(dim=2)

                with self.assertRaisesRegex(
                    ValueError,
                    "noise_weights must be scalar or one-dimensional",
                ):
                    filter_instance.predict_nonlinear_arbitrary_noise(
                        lambda x, _v: x,
                        array([[0.0, 0.0]]),
                        weights,
                    )

                self._assert_initial_state(filter_instance)

    def test_arbitrary_noise_rejects_matrix_transition_outputs(self):
        for output_shape in ((1, 2), (2, 1)):
            with self.subTest(output_shape=output_shape):
                filter_instance = HypersphericalUKF(dim=2)

                with self.assertRaisesRegex(
                    ValueError,
                    "transition function output must be scalar or one-dimensional",
                ):
                    filter_instance.predict_nonlinear_arbitrary_noise(
                        lambda x, _v, shape=output_shape: reshape(x, shape),
                        array([[0.0]]),
                        1.0,
                    )

                self._assert_initial_state(filter_instance)


if __name__ == "__main__":
    unittest.main()
