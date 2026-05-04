import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, diag, eye
from pyrecest.models import (
    AdditiveNoiseMeasurementModel,
    AdditiveNoiseTransitionModel,
)


class DummyNoise:
    def __init__(self, mean, covariance, samples=None, pdf_value=None):
        self._mean = mean
        self._covariance = covariance
        self._samples = samples
        self._pdf_value = pdf_value
        self.last_pdf_arg = None

    def mean(self):
        return self._mean

    def covariance(self):
        return self._covariance

    def sample(self, n):
        if self._samples is None:
            raise AssertionError("No samples configured")
        return self._samples[:n]

    def pdf(self, x):
        self.last_pdf_arg = x
        return self._pdf_value


def assert_backend_allclose(test_case, actual, expected):
    test_case.assertTrue(bool(allclose(actual, expected)))


class AdditiveNoiseTransitionModelTest(unittest.TestCase):
    def test_transition_model_exposes_function_statistics_samples_and_jacobian(self):
        noise = DummyNoise(
            mean=array([0.1, -0.2]),
            covariance=diag(array([1.0, 2.0])),
            samples=array([[0.1, -0.2], [0.3, 0.4]]),
        )
        model = AdditiveNoiseTransitionModel(
            lambda x: array([x[0] + 1.0, 2.0 * x[1]]),
            noise_distribution=noise,
            jacobian=lambda _x: diag(array([1.0, 2.0])),
        )

        state = array([2.0, 3.0])

        assert_backend_allclose(
            self, model.transition_function(state), array([3.0, 6.0])
        )
        assert_backend_allclose(self, model.mean(state), array([3.1, 5.8]))
        assert_backend_allclose(self, model.noise_covariance, diag(array([1.0, 2.0])))
        assert_backend_allclose(self, model.jacobian(state), diag(array([1.0, 2.0])))
        assert_backend_allclose(
            self,
            model.sample_next(state, n=2),
            array([[3.1, 5.8], [3.3, 6.4]]),
        )
        self.assertTrue(model.has_jacobian())

    def test_transition_density_uses_additive_residual(self):
        noise = DummyNoise(
            mean=array([0.0]),
            covariance=array([[1.0]]),
            pdf_value=array(7.0),
        )
        model = AdditiveNoiseTransitionModel(
            lambda x: array([x[0] + 1.0]), noise_distribution=noise
        )

        likelihood = model.transition_density(array([4.5]), array([2.0]))

        assert_backend_allclose(self, likelihood, array(7.0))
        assert_backend_allclose(self, noise.last_pdf_arg, array([1.5]))

    def test_explicit_noise_statistics_work_without_distribution(self):
        model = AdditiveNoiseTransitionModel(
            lambda x: x,
            noise_mean=array([1.0, -1.0]),
            noise_covariance=eye(2),
        )

        assert_backend_allclose(self, model.mean(array([2.0, 3.0])), array([3.0, 2.0]))
        assert_backend_allclose(self, model.noise_covariance, eye(2))

    def test_missing_transition_capabilities_raise(self):
        model = AdditiveNoiseTransitionModel(lambda x: x)

        with self.assertRaises(NotImplementedError):
            model.jacobian(array([1.0]))
        with self.assertRaises(NotImplementedError):
            model.sample_next(array([1.0]))
        with self.assertRaises(NotImplementedError):
            model.transition_density(array([1.0]), array([1.0]))


class AdditiveNoiseMeasurementModelTest(unittest.TestCase):
    def test_measurement_model_exposes_prediction_statistics_samples_and_jacobian(self):
        noise = DummyNoise(
            mean=array([0.5]),
            covariance=array([[2.0]]),
            samples=array([[0.5], [-0.5]]),
        )
        model = AdditiveNoiseMeasurementModel(
            lambda x: array([x[0] * x[0]]),
            noise_distribution=noise,
            jacobian=lambda x: array([[2.0 * x[0]]]),
        )

        state = array([2.0])

        assert_backend_allclose(self, model.measurement_function(state), array([4.0]))
        assert_backend_allclose(self, model.predict_measurement(state), array([4.5]))
        assert_backend_allclose(self, model.mean(state), array([4.5]))
        assert_backend_allclose(self, model.noise_covariance, array([[2.0]]))
        assert_backend_allclose(self, model.jacobian(state), array([[4.0]]))
        assert_backend_allclose(
            self, model.sample_measurement(state, n=2), array([[4.5], [3.5]])
        )
        self.assertTrue(model.has_jacobian())

    def test_likelihood_uses_measurement_residual(self):
        noise = DummyNoise(
            mean=array([0.0]),
            covariance=array([[1.0]]),
            pdf_value=array(11.0),
        )
        model = AdditiveNoiseMeasurementModel(
            lambda x: array([x[0] * x[0]]), noise_distribution=noise
        )

        likelihood = model.likelihood(array([5.0]), array([2.0]))

        assert_backend_allclose(self, likelihood, array(11.0))
        assert_backend_allclose(
            self, model.measurement_residual(array([5.0]), array([2.0])), array([1.0])
        )
        assert_backend_allclose(self, noise.last_pdf_arg, array([1.0]))

    def test_missing_measurement_capabilities_raise(self):
        model = AdditiveNoiseMeasurementModel(lambda x: x)

        with self.assertRaises(NotImplementedError):
            model.jacobian(array([1.0]))
        with self.assertRaises(NotImplementedError):
            model.sample_measurement(array([1.0]))
        with self.assertRaises(NotImplementedError):
            model.likelihood(array([1.0]), array([1.0]))

    def test_constructor_rejects_non_callables(self):
        with self.assertRaises(TypeError):
            AdditiveNoiseTransitionModel(1)
        with self.assertRaises(TypeError):
            AdditiveNoiseMeasurementModel(1)
        with self.assertRaises(TypeError):
            AdditiveNoiseTransitionModel(lambda x: x, jacobian=1)
        with self.assertRaises(TypeError):
            AdditiveNoiseMeasurementModel(lambda x: x, jacobian=1)


if __name__ == "__main__":
    unittest.main()
