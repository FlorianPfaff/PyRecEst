import unittest

from pyrecest.backend import array, diag, eye
from pyrecest.models import (
    IdentityGaussianMeasurementModel,
    IdentityGaussianTransitionModel,
    LinearGaussianMeasurementModel,
    LinearGaussianTransitionModel,
)


class LinearGaussianCovarianceValidationTest(unittest.TestCase):
    def test_rejects_asymmetric_noise_covariances(self):
        covariance = array([[1.0, 0.5], [0.0, 1.0]])

        with self.assertRaises(ValueError):
            LinearGaussianTransitionModel(eye(2), covariance)
        with self.assertRaises(ValueError):
            LinearGaussianMeasurementModel(eye(2), covariance)

    def test_rejects_indefinite_noise_covariances(self):
        covariance = diag(array([1.0, -0.25]))

        with self.assertRaises(ValueError):
            LinearGaussianTransitionModel(eye(2), covariance)
        with self.assertRaises(ValueError):
            LinearGaussianMeasurementModel(eye(2), covariance)

    def test_identity_models_reject_negative_scalar_variance(self):
        with self.assertRaises(ValueError):
            IdentityGaussianTransitionModel(2, -0.25)
        with self.assertRaises(ValueError):
            IdentityGaussianMeasurementModel(2, -0.25)

    def test_prediction_rejects_invalid_state_covariance(self):
        transition = LinearGaussianTransitionModel(eye(2), eye(2))
        measurement = LinearGaussianMeasurementModel(eye(2), eye(2))
        covariance = diag(array([1.0, -0.25]))

        with self.assertRaises(ValueError):
            transition.predict_covariance(covariance)
        with self.assertRaises(ValueError):
            measurement.innovation_covariance(covariance)

    def test_accepts_singular_positive_semidefinite_covariances(self):
        covariance = diag(array([1.0, 0.0]))
        transition = LinearGaussianTransitionModel(eye(2), covariance)
        measurement = LinearGaussianMeasurementModel(eye(2), covariance)

        transition.predict_covariance(covariance)
        measurement.innovation_covariance(covariance)


if __name__ == "__main__":
    unittest.main()
