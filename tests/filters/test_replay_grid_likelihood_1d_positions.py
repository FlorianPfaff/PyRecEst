import unittest

import numpy as np
from pyrecest.filters import (
    particle_position_log_posterior,
    replay_grid_log_likelihood_values,
)


class TestReplayGridLikelihoodOneDimensionalPositions(unittest.TestCase):
    def test_log_likelihood_accepts_vector_of_scalar_positions(self):
        bin_centers = np.asarray([[0.0], [1.0], [2.0]])
        values = np.asarray([0.0, 1.0, 2.0])

        result = replay_grid_log_likelihood_values(
            [0.2, 1.8],
            values,
            bin_centers,
            interpolation="nearest",
        )

        self.assertTrue(np.allclose(result, [0.0, 2.0]))

    def test_log_likelihood_accepts_scalar_position(self):
        bin_centers = np.asarray([[0.0], [1.0], [2.0]])
        values = np.asarray([0.0, 1.0, 2.0])

        result = replay_grid_log_likelihood_values(
            0.9,
            values,
            bin_centers,
            interpolation="nearest",
        )

        self.assertTrue(np.allclose(result, [1.0]))

    def test_particle_position_log_posterior_accepts_1d_position_vector(self):
        bin_centers = np.asarray([[0.0], [1.0]])
        weights = np.asarray([0.25, 0.75])

        log_posterior = particle_position_log_posterior(
            [0.1, 0.9],
            weights,
            bin_centers,
        )

        self.assertTrue(np.allclose(np.exp(log_posterior), [0.25, 0.75]))


if __name__ == "__main__":
    unittest.main()
