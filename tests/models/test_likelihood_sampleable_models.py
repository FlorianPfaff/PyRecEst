import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, arange, array, exp, reshape
from pyrecest.models import (
    DensityTransitionModel,
    LikelihoodMeasurementModel,
    SampleableTransitionModel,
    SupportsLikelihood,
    SupportsLogLikelihood,
    SupportsTransitionDensity,
    SupportsTransitionSampling,
)


class DummyDistribution:
    def __init__(self, offset):
        self.offset = offset

    def pdf(self, measurement):
        return measurement + self.offset

    def log_pdf(self, measurement):
        return measurement - self.offset


class LikelihoodMeasurementModelTest(unittest.TestCase):
    def test_likelihood_callback(self):
        model = LikelihoodMeasurementModel(
            lambda measurement, state: exp(-0.5 * (measurement - state) ** 2),
            name="unit-test-likelihood",
        )

        measurement = array([1.0, 2.0])
        state = array([1.0, 0.0])

        expected = exp(-0.5 * (measurement - state) ** 2)

        self.assertIsInstance(model, SupportsLikelihood)
        self.assertEqual(model.name, "unit-test-likelihood")
        self.assertTrue(allclose(model.likelihood(measurement, state), expected))

    def test_log_likelihood_callback(self):
        model = LikelihoodMeasurementModel(
            lambda measurement, state: measurement + state,
            log_likelihood=lambda measurement, state: measurement - state,
        )

        self.assertIsInstance(model, SupportsLogLikelihood)
        self.assertTrue(model.has_log_likelihood)
        self.assertTrue(
            allclose(model.log_likelihood(array([3.0]), array([1.0])), array([2.0]))
        )

    def test_missing_log_likelihood_raises(self):
        model = LikelihoodMeasurementModel(
            lambda measurement, state: measurement + state
        )

        self.assertFalse(model.has_log_likelihood)
        with self.assertRaises(NotImplementedError):
            model.log_likelihood(array([1.0]), array([0.0]))

    def test_from_distribution_factory(self):
        model = LikelihoodMeasurementModel.from_distribution_factory(
            DummyDistribution,
            log_pdf_method="log_pdf",
        )

        self.assertTrue(
            allclose(model.likelihood(array([2.0]), array([3.0])), array([5.0]))
        )
        self.assertTrue(
            allclose(model.log_likelihood(array([2.0]), array([3.0])), array([-1.0]))
        )

    def test_non_callable_likelihood_rejected(self):
        with self.assertRaises(TypeError):
            LikelihoodMeasurementModel(None)


class SampleableTransitionModelTest(unittest.TestCase):
    def test_sample_next_callback(self):
        def sample_next(state, n=1):
            return state + reshape(arange(n), (n, 1))

        model = SampleableTransitionModel(sample_next, name="unit-test-transition")

        self.assertIsInstance(model, SupportsTransitionSampling)
        self.assertEqual(model.name, "unit-test-transition")
        self.assertFalse(model.has_transition_density)
        self.assertTrue(
            allclose(
                model.sample_next(array([10.0]), n=3), array([[10.0], [11.0], [12.0]])
            )
        )

    def test_unary_sample_next_callback(self):
        def shift_state(state):
            return state + array([1.0])

        model = SampleableTransitionModel(shift_state)

        self.assertTrue(allclose(model.sample_next(array([10.0])), array([11.0])))

    def test_optional_transition_density(self):
        model = SampleableTransitionModel(
            lambda state, n=1: state + reshape(arange(n), (n, 1)),
            transition_density=lambda state_next, state_previous: exp(
                -0.5 * (state_next - state_previous) ** 2
            ),
        )

        self.assertTrue(model.has_transition_density)
        self.assertTrue(
            allclose(model.transition_density(array([1.0]), array([1.0])), array([1.0]))
        )

    def test_missing_transition_density_raises(self):
        model = SampleableTransitionModel(lambda state, n=1: state)

        with self.assertRaises(NotImplementedError):
            model.transition_density(array([1.0]), array([0.0]))

    def test_particle_filter_vectorization_flag_is_stored(self):
        def identity_state(state):
            return state

        model = SampleableTransitionModel(
            identity_state,
            function_is_vectorized=False,
        )

        self.assertFalse(model.function_is_vectorized)


class DensityTransitionModelTest(unittest.TestCase):
    def test_transition_density_callback(self):
        model = DensityTransitionModel(
            lambda state_next, state_previous: exp(
                -0.5 * (state_next - state_previous) ** 2
            )
        )

        self.assertIsInstance(model, SupportsTransitionDensity)
        self.assertFalse(model.has_sampler)
        self.assertTrue(
            allclose(model.transition_density(array([2.0]), array([2.0])), array([1.0]))
        )

    def test_optional_sampler(self):
        def sample_next(state, n=1):
            return state + reshape(arange(n), (n, 1))

        model = DensityTransitionModel(
            lambda state_next, state_previous: exp(
                -0.5 * (state_next - state_previous) ** 2
            ),
            sample_next=sample_next,
        )

        self.assertTrue(model.has_sampler)
        self.assertTrue(
            allclose(model.sample_next(array([4.0]), n=2), array([[4.0], [5.0]]))
        )

    def test_missing_sampler_raises(self):
        model = DensityTransitionModel(
            lambda state_next, state_previous: state_next + state_previous
        )

        with self.assertRaises(NotImplementedError):
            model.sample_next(array([1.0]))


if __name__ == "__main__":
    unittest.main()
