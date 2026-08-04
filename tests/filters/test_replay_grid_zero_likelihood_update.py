"""Regression tests for impossible replay-grid update likelihoods."""

import types
import unittest

import numpy as np

from pyrecest.filters import update_position_grid_likelihood


class _RecordingLikelihoodFilter:
    def __init__(self, positions):
        self.position_particles = np.asarray(positions, dtype=float)
        self.filter_state = types.SimpleNamespace(
            w=np.full(
                self.position_particles.shape[0],
                1.0 / self.position_particles.shape[0],
            )
        )
        self.last_likelihood_values = None

    def update_position_likelihood(self, likelihood, *, return_log_marginal=False):
        self.last_likelihood_values = np.asarray(
            likelihood(self.position_particles), dtype=float
        )
        marginal = float(
            np.average(self.last_likelihood_values, weights=self.filter_state.w)
        )
        if return_log_marginal:
            return float(np.log(marginal))
        return self


class TestReplayGridZeroLikelihoodUpdate(unittest.TestCase):
    def test_update_preserves_exact_zero_for_impossible_grid_bins(self):
        bin_centers = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        replay_filter = _RecordingLikelihoodFilter(bin_centers)

        log_marginal = update_position_grid_likelihood(
            replay_filter,
            np.asarray([float("-inf"), 0.0]),
            bin_centers,
            interpolation="nearest",
        )

        np.testing.assert_array_equal(
            replay_filter.last_likelihood_values,
            np.asarray([0.0, 1.0]),
        )
        self.assertAlmostEqual(log_marginal, np.log(0.5))


if __name__ == "__main__":
    unittest.main()
