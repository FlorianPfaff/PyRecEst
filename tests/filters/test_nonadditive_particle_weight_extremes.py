import unittest

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, float64
from pyrecest.distributions import LinearDiracDistribution
from pyrecest.filters.abstract_particle_filter import (
    _normalize_nonadditive_noise_weights,
)
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter


class NonadditiveParticleWeightExtremesTest(unittest.TestCase):
    @staticmethod
    def _subnormal_weights():
        tiny_weight = np.finfo(np.float64).tiny / 1024.0
        return array([2.0 * tiny_weight, tiny_weight], dtype=float64)

    @staticmethod
    def _overflowing_sum_weights():
        max_float = np.finfo(np.float64).max
        return array([max_float, 0.5 * max_float], dtype=float64)

    def test_extreme_valid_weights_preserve_ratios(self):
        expected = array([2.0 / 3.0, 1.0 / 3.0], dtype=float64)
        for weights in (
            self._subnormal_weights(),
            self._overflowing_sum_weights(),
        ):
            with self.subTest(weights=weights):
                normalized = _normalize_nonadditive_noise_weights(weights)
                self.assertTrue(allclose(normalized, expected))

    def test_negative_subnormal_weight_is_rejected(self):
        tiny_weight = np.finfo(np.float64).tiny / 1024.0
        weights = array([1.0, -tiny_weight], dtype=float64)

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _normalize_nonadditive_noise_weights(weights)

    def test_public_prediction_accepts_extreme_valid_weights(self):
        samples = array([[0.0], [0.0]])
        initial_particles = array([[0.0], [1.0], [2.0], [3.0]])

        for weights in (
            self._subnormal_weights(),
            self._overflowing_sum_weights(),
        ):
            with self.subTest(weights=weights):
                particle_filter = EuclideanParticleFilter(n_particles=4, dim=1)
                particle_filter.filter_state = LinearDiracDistribution(initial_particles)
                particle_filter.predict_nonlinear_nonadditive(
                    lambda particle, noise: particle + noise,
                    samples,
                    weights,
                )
                self.assertTrue(allclose(particle_filter.filter_state.d, initial_particles))


if __name__ == "__main__":
    unittest.main()
