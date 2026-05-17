import copy
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, random
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter
from pyrecest.models import (
    AdditiveNoiseTransitionModel,
    LikelihoodMeasurementModel,
    SampleableTransitionModel,
)


class RecordingNoise:
    def __init__(self, samples):
        self.samples = samples
        self.calls = []

    def sample(self, n):
        start = sum(self.calls)
        self.calls.append(n)
        return self.samples[start : start + n]


class ParticleFilterModelAdapterTest(unittest.TestCase):
    def setUp(self):
        self.particle_locations = array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )
        self.filter = EuclideanParticleFilter(n_particles=4, dim=2)
        self.filter.filter_state = LinearDiracDistribution(self.particle_locations)

    def test_predict_model_with_vectorized_sampleable_transition(self):
        direct_filter = copy.deepcopy(self.filter)
        model_filter = copy.deepcopy(self.filter)
        shift = array([1.0, -2.0])

        direct_filter.predict_nonlinear(
            lambda particles: particles + shift,
            noise_distribution=None,
            function_is_vectorized=True,
        )
        model_filter.predict_model(
            SampleableTransitionModel(lambda particles: particles + shift)
        )

        npt.assert_allclose(model_filter.filter_state.d, direct_filter.filter_state.d)
        npt.assert_allclose(model_filter.filter_state.w, direct_filter.filter_state.w)

    def test_predict_model_with_nonvectorized_sampleable_transition(self):
        direct_filter = copy.deepcopy(self.filter)
        model_filter = copy.deepcopy(self.filter)
        shift = array([1.0, -2.0])

        direct_filter.predict_nonlinear(
            lambda particle: particle + shift,
            noise_distribution=None,
            function_is_vectorized=False,
        )
        model_filter.predict_model(
            SampleableTransitionModel(
                lambda particle: particle + shift,
                function_is_vectorized=False,
            )
        )

        npt.assert_allclose(model_filter.filter_state.d, direct_filter.filter_state.d)
        npt.assert_allclose(model_filter.filter_state.w, direct_filter.filter_state.w)

    def test_predict_model_passes_particle_count_to_vectorized_additive_transition(
        self,
    ):
        model_filter = copy.deepcopy(self.filter)
        noise_samples = array(
            [
                [10.0, 0.0],
                [20.0, 0.0],
                [30.0, 0.0],
                [40.0, 0.0],
            ]
        )
        noise = RecordingNoise(noise_samples)

        model_filter.predict_model(
            AdditiveNoiseTransitionModel(
                lambda particles: particles,
                noise_distribution=noise,
                vectorized=True,
            )
        )

        npt.assert_allclose(
            model_filter.filter_state.d, self.particle_locations + noise_samples
        )
        self.assertEqual(noise.calls, [4])

    def test_predict_model_honors_additive_transition_vectorized_flag(self):
        model_filter = copy.deepcopy(self.filter)
        noise_samples = array(
            [
                [10.0, 0.0],
                [20.0, 0.0],
                [30.0, 0.0],
                [40.0, 0.0],
            ]
        )
        noise = RecordingNoise(noise_samples)

        model_filter.predict_model(
            AdditiveNoiseTransitionModel(
                lambda particle: particle,
                noise_distribution=noise,
                vectorized=False,
            )
        )

        npt.assert_allclose(
            model_filter.filter_state.d, self.particle_locations + noise_samples
        )
        self.assertEqual(noise.calls, [1, 1, 1, 1])

    def test_update_model_with_likelihood_measurement_model(self):
        measurement = array([2.0])

        def likelihood(meas, particles):
            residual = particles[:, 0] - meas[0]
            return 1.0 * (residual == 0.0)

        direct_filter = copy.deepcopy(self.filter)
        model_filter = copy.deepcopy(self.filter)

        random.seed(7)
        direct_filter.update_nonlinear_using_likelihood(likelihood, measurement)
        random.seed(7)
        model_filter.update_model(LikelihoodMeasurementModel(likelihood), measurement)

        npt.assert_allclose(model_filter.filter_state.d, direct_filter.filter_state.d)
        npt.assert_allclose(model_filter.filter_state.w, direct_filter.filter_state.w)


if __name__ == "__main__":
    unittest.main()
