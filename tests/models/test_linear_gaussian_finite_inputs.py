import unittest

import numpy as np
from pyrecest.backend import array
from pyrecest.models import (
    IdentityGaussianMeasurementModel,
    IdentityGaussianTransitionModel,
    LinearGaussianMeasurementModel,
    LinearGaussianTransitionModel,
)


class LinearGaussianFiniteInputsTest(unittest.TestCase):
    def test_models_reject_nonfinite_system_and_measurement_matrices(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(model="transition", value=value):
                with self.assertRaisesRegex(ValueError, "matrix.*finite"):
                    LinearGaussianTransitionModel(array([[value]]), array([[1.0]]))
            with self.subTest(model="measurement", value=value):
                with self.assertRaisesRegex(ValueError, "matrix.*finite"):
                    LinearGaussianMeasurementModel(array([[value]]), array([[1.0]]))

    def test_models_reject_nonfinite_noise_covariances(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(model="transition", value=value):
                with self.assertRaisesRegex(ValueError, "noise_cov.*finite"):
                    LinearGaussianTransitionModel(array([[1.0]]), array([[value]]))
            with self.subTest(model="measurement", value=value):
                with self.assertRaisesRegex(ValueError, "noise_cov.*finite"):
                    LinearGaussianMeasurementModel(array([[1.0]]), array([[value]]))

    def test_transition_model_rejects_nonfinite_offset(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "offset.*finite"):
                    LinearGaussianTransitionModel(
                        array([[1.0]]), array([[1.0]]), offset=array([value])
                    )

    def test_identity_models_reject_nonfinite_scalar_noise(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(model="transition", value=value):
                with self.assertRaisesRegex(ValueError, "noise_cov.*finite"):
                    IdentityGaussianTransitionModel(1, value)
            with self.subTest(model="measurement", value=value):
                with self.assertRaisesRegex(ValueError, "noise_cov.*finite"):
                    IdentityGaussianMeasurementModel(1, value)

    def test_prediction_rejects_nonfinite_state_inputs(self):
        transition = LinearGaussianTransitionModel(array([[1.0]]), array([[1.0]]))
        measurement = LinearGaussianMeasurementModel(array([[1.0]]), array([[1.0]]))

        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(method="transition mean", value=value):
                with self.assertRaisesRegex(ValueError, "state_mean.*finite"):
                    transition.predict_mean(array([value]))
            with self.subTest(method="measurement mean", value=value):
                with self.assertRaisesRegex(ValueError, "state_mean.*finite"):
                    measurement.predict_mean(array([value]))
            with self.subTest(method="transition covariance", value=value):
                with self.assertRaisesRegex(ValueError, "state_covariance.*finite"):
                    transition.predict_covariance(array([[value]]))
            with self.subTest(method="measurement covariance", value=value):
                with self.assertRaisesRegex(ValueError, "state_covariance.*finite"):
                    measurement.innovation_covariance(array([[value]]))


if __name__ == "__main__":
    unittest.main()
