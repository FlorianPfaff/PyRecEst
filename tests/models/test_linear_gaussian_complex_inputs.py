import unittest

import numpy as np
from pyrecest.backend import array, eye
from pyrecest.models import (
    LinearGaussianMeasurementModel,
    LinearGaussianTransitionModel,
)


class LinearGaussianComplexInputTest(unittest.TestCase):
    def test_transition_model_rejects_complex_parameters(self):
        complex_matrix = np.asarray([[1.0 + 1.0j]])
        real_matrix = np.asarray([[1.0]])
        complex_vector = np.asarray([1.0 + 1.0j])

        invalid_factories = (
            lambda: LinearGaussianTransitionModel(complex_matrix, real_matrix),
            lambda: LinearGaussianTransitionModel(real_matrix, complex_matrix),
            lambda: LinearGaussianTransitionModel(
                real_matrix, real_matrix, offset=complex_vector
            ),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory):
                with self.assertRaisesRegex(ValueError, "real-valued"):
                    factory()

    def test_prediction_rejects_complex_state_inputs(self):
        model = LinearGaussianTransitionModel(eye(1), array([[1.0]]))

        with self.assertRaisesRegex(ValueError, "state_mean must be real-valued"):
            model.predict_mean(np.asarray([1.0 + 1.0j]))
        with self.assertRaisesRegex(ValueError, "state_covariance must be real-valued"):
            model.predict_covariance(np.asarray([[1.0 + 1.0j]]))

    def test_measurement_model_rejects_complex_inputs(self):
        real_matrix = np.asarray([[1.0]])
        complex_matrix = np.asarray([[1.0 + 1.0j]])

        with self.assertRaisesRegex(ValueError, "matrix must be real-valued"):
            LinearGaussianMeasurementModel(complex_matrix, real_matrix)
        model = LinearGaussianMeasurementModel(real_matrix, real_matrix)
        with self.assertRaisesRegex(ValueError, "state_mean must be real-valued"):
            model.predict_mean(np.asarray([1.0 + 1.0j]))
        with self.assertRaisesRegex(ValueError, "state_covariance must be real-valued"):
            model.innovation_covariance(np.asarray([[1.0 + 1.0j]]))


if __name__ == "__main__":
    unittest.main()
