import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, reshape, to_numpy
from pyrecest.distributions import LinearDiracDistribution
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter


class _FailingShiftNoise:
    dim = 2

    def set_mean(self, mean):
        mean_numpy = np.asarray(to_numpy(mean), dtype=float)
        if mean_numpy[0] >= 2.0:
            raise RuntimeError("synthetic noise failure")
        self._mean = array(mean_numpy)
        return self

    def sample(self, n):
        if n != 1:
            raise ValueError("test noise only supports one sample")
        return reshape(self._mean, (1, -1))


class ParticleFilterPredictionAtomicityTest(unittest.TestCase):
    @staticmethod
    def _make_filter():
        particles = array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        filter_ = EuclideanParticleFilter(n_particles=3, dim=2)
        filter_.filter_state = LinearDiracDistribution(particles)
        return filter_, particles

    def test_vectorized_prediction_rejects_shape_change_without_mutating_state(self):
        filter_, particles = self._make_filter()

        with self.assertRaisesRegex(ValueError, "Prediction function returned particles"):
            filter_.predict_nonlinear(
                lambda particle_matrix: particle_matrix[:, 0],
                function_is_vectorized=True,
            )

        npt.assert_allclose(to_numpy(filter_.filter_state.d), to_numpy(particles))

    def test_nonvectorized_prediction_rolls_back_when_noise_sampling_fails(self):
        filter_, particles = self._make_filter()

        with self.assertRaisesRegex(RuntimeError, "synthetic noise failure"):
            filter_.predict_nonlinear(
                lambda particle: particle + 1.0,
                noise_distribution=_FailingShiftNoise(),
                function_is_vectorized=False,
            )

        npt.assert_allclose(to_numpy(filter_.filter_state.d), to_numpy(particles))


if __name__ == "__main__":
    unittest.main()
