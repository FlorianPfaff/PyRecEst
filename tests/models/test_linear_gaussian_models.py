import unittest

import numpy.testing as npt
from pyrecest.backend import array, diag, eye, to_numpy
from pyrecest.distributions import GaussianDistribution
from pyrecest.models import (
    IdentityGaussianMeasurementModel,
    IdentityGaussianTransitionModel,
    LinearGaussianMeasurementModel,
    LinearGaussianTransitionModel,
)


class LinearGaussianModelsTest(unittest.TestCase):
    def test_transition_predict_distribution(self):
        model = LinearGaussianTransitionModel(
            array([[1.0, 1.0], [0.0, 1.0]]),
            diag(array([0.1, 0.2])),
            array([0.5, -0.25]),
        )
        state = GaussianDistribution(array([2.0, 0.5]), diag(array([1.0, 2.0])))
        predicted = model.predict_distribution(state)

        npt.assert_allclose(
            to_numpy(predicted.mu), to_numpy(array([3.0, 0.25])), atol=1e-12
        )
        npt.assert_allclose(
            to_numpy(predicted.C), to_numpy(array([[3.1, 2.0], [2.0, 2.2]])), atol=1e-12
        )

    def test_transition_noise_distribution_and_alias(self):
        noise_cov = diag(array([0.1, 0.2]))
        model = LinearGaussianTransitionModel(eye(2), noise_cov)

        self.assertIs(model.system_noise_cov, model.sys_noise_cov)
        noise = model.noise_distribution()

        npt.assert_allclose(to_numpy(noise.mu), to_numpy(array([0.0, 0.0])), atol=1e-12)
        npt.assert_allclose(to_numpy(noise.C), to_numpy(noise_cov), atol=1e-12)

    def test_measurement_predict_distribution(self):
        model = LinearGaussianMeasurementModel(array([[1.0, 0.0]]), array([[0.25]]))
        state = GaussianDistribution(array([2.0, 0.5]), diag(array([1.0, 2.0])))
        predicted = model.predict_distribution(state)

        npt.assert_allclose(to_numpy(predicted.mu), to_numpy(array([2.0])), atol=1e-12)
        npt.assert_allclose(
            to_numpy(predicted.C), to_numpy(array([[1.25]])), atol=1e-12
        )

    def test_measurement_noise_distribution_and_alias(self):
        noise_cov = array([[0.25]])
        model = LinearGaussianMeasurementModel(array([[1.0, 0.0]]), noise_cov)

        self.assertIs(model.measurement_noise_cov, model.meas_noise)
        noise = model.noise_distribution()

        npt.assert_allclose(to_numpy(noise.mu), to_numpy(array([0.0])), atol=1e-12)
        npt.assert_allclose(to_numpy(noise.C), to_numpy(noise_cov), atol=1e-12)

    def test_identity_models(self):
        transition_model = IdentityGaussianTransitionModel(2, diag(array([0.1, 0.2])))
        measurement_model = IdentityGaussianMeasurementModel(2, diag(array([0.3, 0.4])))

        npt.assert_allclose(
            to_numpy(transition_model.system_matrix), to_numpy(eye(2)), atol=1e-12
        )
        npt.assert_allclose(
            to_numpy(measurement_model.measurement_matrix), to_numpy(eye(2)), atol=1e-12
        )

    def test_rejects_incompatible_shapes(self):
        with self.assertRaises(ValueError):
            LinearGaussianTransitionModel(array([[1.0, 0.0]]), diag(array([0.1, 0.2])))
        with self.assertRaises(ValueError):
            LinearGaussianMeasurementModel(array([[1.0, 0.0]]), diag(array([0.1, 0.2])))

        transition_model = LinearGaussianTransitionModel(
            eye(2), diag(array([0.1, 0.2]))
        )
        with self.assertRaises(ValueError):
            transition_model.predict_mean(array([1.0, 2.0, 3.0]))

        measurement_model = LinearGaussianMeasurementModel(
            array([[1.0, 0.0]]), array([[0.25]])
        )
        with self.assertRaises(ValueError):
            measurement_model.predict_mean(array([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
