"""Tests for protocol-based model adapter helpers."""

import unittest

# pylint: disable=no-name-in-module,no-member,too-few-public-methods
from pyrecest.backend import allclose, arange, array, diag, exp, reshape
from pyrecest.models import (
    DensityTransitionModel,
    LikelihoodMeasurementModel,
    LinearGaussianMeasurementModel,
    LinearGaussianTransitionModel,
    SampleableTransitionModel,
    as_density_transition_model,
    as_likelihood_model,
    as_sampleable_transition_model,
    evaluate_likelihood,
    evaluate_log_likelihood,
    evaluate_transition_density,
    get_optional_model_attribute,
    linear_measurement_arguments,
    linear_transition_arguments,
    predict_distribution_from_model,
    require_model_attribute,
    sample_next_state,
)
from pyrecest.protocols.models import (
    SupportsLikelihood,
    SupportsLinearGaussianMeasurement,
    SupportsLinearGaussianTransition,
    SupportsLogLikelihood,
    SupportsPredictedDistribution,
    SupportsTransitionDensity,
    SupportsTransitionSampling,
)


class DistributionPredictor:
    def predict_distribution(self, state_distribution, scale=1):
        return state_distribution * scale


class AliasTransitionModel:
    def __init__(self, system_matrix, sys_noise_cov, system_input=None):
        self.system_matrix = system_matrix
        self.sys_noise_cov = sys_noise_cov
        if system_input is not None:
            self.system_input = system_input


class AliasMeasurementModel:
    def __init__(self, measurement_matrix, meas_noise):
        self.measurement_matrix = measurement_matrix
        self.meas_noise = meas_noise


class ModelAdapterTest(unittest.TestCase):
    def test_as_likelihood_model_wraps_callback(self):
        def likelihood(measurement, state):
            return measurement + state

        model = as_likelihood_model(likelihood, name="wrapped-likelihood")

        self.assertIsInstance(model, LikelihoodMeasurementModel)
        self.assertIsInstance(model, SupportsLikelihood)
        self.assertEqual(model.name, "wrapped-likelihood")
        self.assertTrue(
            allclose(
                evaluate_likelihood(model, array([2.0]), array([3.0])),
                array([5.0]),
            )
        )

    def test_as_likelihood_model_returns_existing_model(self):
        def likelihood(measurement, state):
            return measurement + state

        model = LikelihoodMeasurementModel(likelihood)

        self.assertIs(as_likelihood_model(model), model)

    def test_as_likelihood_model_rejects_metadata_for_existing_model(self):
        def likelihood(measurement, state):
            return measurement + state

        model = LikelihoodMeasurementModel(likelihood)

        with self.assertRaises(ValueError):
            as_likelihood_model(model, name="ignored")

    def test_evaluate_log_likelihood(self):
        def likelihood(measurement, state):
            return exp(-(measurement - state) ** 2)

        def log_likelihood(measurement, state):
            return -(measurement - state) ** 2

        model = as_likelihood_model(
            likelihood,
            log_likelihood=log_likelihood,
        )

        self.assertIsInstance(model, SupportsLogLikelihood)
        self.assertTrue(
            allclose(
                evaluate_log_likelihood(model, array([3.0]), array([1.0])),
                array([-4.0]),
            )
        )

    def test_evaluate_likelihood_rejects_missing_capability(self):
        with self.assertRaises(TypeError):
            evaluate_likelihood(object(), array([1.0]), array([0.0]))

    def test_as_sampleable_transition_model_wraps_callback(self):
        def sample_next(state, n=1):
            return state + reshape(arange(n), (n, 1))

        model = as_sampleable_transition_model(
            sample_next,
            name="wrapped-transition",
        )

        self.assertIsInstance(model, SampleableTransitionModel)
        self.assertIsInstance(model, SupportsTransitionSampling)
        self.assertEqual(model.name, "wrapped-transition")
        self.assertTrue(
            allclose(
                sample_next_state(model, array([10.0]), n=3),
                array([[10.0], [11.0], [12.0]]),
            )
        )

    def test_as_sampleable_transition_model_returns_existing_model(self):
        def sample_next(state, n=1):
            return state + reshape(arange(n), (n, 1))

        model = SampleableTransitionModel(sample_next)

        self.assertIs(as_sampleable_transition_model(model), model)

    def test_as_density_transition_model_wraps_callback(self):
        def transition_density(state_next, state_previous):
            return exp(-0.5 * (state_next - state_previous) ** 2)

        model = as_density_transition_model(
            transition_density,
            name="wrapped-density",
        )

        self.assertIsInstance(model, DensityTransitionModel)
        self.assertIsInstance(model, SupportsTransitionDensity)
        self.assertEqual(model.name, "wrapped-density")
        self.assertTrue(
            allclose(
                evaluate_transition_density(model, array([1.0]), array([1.0])),
                array([1.0]),
            )
        )

    def test_predict_distribution_from_model(self):
        model = DistributionPredictor()

        self.assertIsInstance(model, SupportsPredictedDistribution)
        self.assertTrue(
            allclose(
                predict_distribution_from_model(model, array([2.0]), scale=3),
                array([6.0]),
            )
        )

    def test_predict_distribution_rejects_missing_capability(self):
        with self.assertRaises(TypeError):
            predict_distribution_from_model(object(), array([1.0]))

    def test_linear_transition_arguments_accept_canonical_model(self):
        system_matrix = diag(array([1.0, 2.0]))
        noise_cov = diag(array([0.1, 0.2]))
        sys_input = array([3.0, 4.0])
        model = LinearGaussianTransitionModel(system_matrix, noise_cov, sys_input)

        self.assertIsInstance(model, SupportsLinearGaussianTransition)
        args = linear_transition_arguments(model)

        self.assertTrue(allclose(args.system_matrix, system_matrix))
        self.assertTrue(allclose(args.sys_noise_cov, noise_cov))
        self.assertTrue(allclose(args.sys_input, sys_input))

    def test_linear_transition_arguments_accept_aliases(self):
        system_matrix = diag(array([1.0, 2.0]))
        noise_cov = diag(array([0.1, 0.2]))
        system_input = array([3.0, 4.0])
        model = AliasTransitionModel(system_matrix, noise_cov, system_input)

        args = linear_transition_arguments(model)

        self.assertTrue(allclose(args.system_matrix, system_matrix))
        self.assertTrue(allclose(args.sys_noise_cov, noise_cov))
        self.assertTrue(allclose(args.sys_input, system_input))

    def test_linear_measurement_arguments_accept_canonical_model(self):
        measurement_matrix = diag(array([1.0, 2.0]))
        noise_cov = diag(array([0.1, 0.2]))
        model = LinearGaussianMeasurementModel(measurement_matrix, noise_cov)

        self.assertIsInstance(model, SupportsLinearGaussianMeasurement)
        args = linear_measurement_arguments(model)

        self.assertTrue(allclose(args.measurement_matrix, measurement_matrix))
        self.assertTrue(allclose(args.meas_noise, noise_cov))

    def test_linear_measurement_arguments_accept_aliases(self):
        measurement_matrix = diag(array([1.0, 2.0]))
        noise_cov = diag(array([0.1, 0.2]))
        model = AliasMeasurementModel(measurement_matrix, noise_cov)

        args = linear_measurement_arguments(model)

        self.assertTrue(allclose(args.measurement_matrix, measurement_matrix))
        self.assertTrue(allclose(args.meas_noise, noise_cov))

    def test_structural_attribute_helpers(self):
        model = AliasTransitionModel(
            diag(array([1.0, 2.0])),
            diag(array([0.1, 0.2])),
        )

        self.assertIs(
            require_model_attribute(model, "system_matrix"),
            model.system_matrix,
        )
        self.assertEqual(
            get_optional_model_attribute(model, "missing", default="fallback"),
            "fallback",
        )

        with self.assertRaises(AttributeError):
            require_model_attribute(model, "missing")


if __name__ == "__main__":
    unittest.main()
