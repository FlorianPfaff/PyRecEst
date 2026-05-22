import types
import unittest

import numpy as np

from pyrecest.filters import (
    adaptive_position_proposal_probability,
    build_replay_grid_likelihood_lookup,
    effective_sample_size_fraction,
    grid_proposal_weights,
    particle_position_log_posterior,
    replay_grid_log_likelihood_values,
    update_position_grid_likelihood,
)


class TestReplayGridLikelihood(unittest.TestCase):
    def test_linear_grid_likelihood_interpolates_rectilinear_log_values(self):
        bin_centers = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        values = 20.0 * bin_centers[:, 0] + 10.0 * bin_centers[:, 1]
        lookup = build_replay_grid_likelihood_lookup(bin_centers, "linear")

        result = replay_grid_log_likelihood_values(
            np.asarray([[0.25, 0.50]]),
            values,
            bin_centers,
            lookup=lookup,
        )

        self.assertEqual(lookup.method, "linear")
        self.assertTrue(np.allclose(result, [10.0]))

    def test_linear_grid_likelihood_falls_back_to_nearest_outside_grid(self):
        bin_centers = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        values = np.asarray([0.0, 1.0, 2.0, 3.0])

        result = replay_grid_log_likelihood_values(
            np.asarray([[2.0, 2.0]]),
            values,
            bin_centers,
            interpolation="linear",
        )

        self.assertTrue(np.allclose(result, [3.0]))

    def test_linear_grid_likelihood_uses_nearest_for_irregular_bins(self):
        bin_centers = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )

        lookup = build_replay_grid_likelihood_lookup(bin_centers, "linear")

        self.assertEqual(lookup.method, "nearest")

    def test_position_proposal_probability_is_ess_adaptive(self):
        self.assertAlmostEqual(effective_sample_size_fraction(np.ones(4)), 1.0)
        probability, ess_fraction = adaptive_position_proposal_probability(
            _DummyFilter(np.ones(4)),
            0.5,
            0.5,
        )
        self.assertEqual(probability, 0.0)
        self.assertAlmostEqual(ess_fraction, 1.0)

        probability, ess_fraction = adaptive_position_proposal_probability(
            _DummyFilter([1.0, 0.0, 0.0, 0.0]),
            0.5,
            0.5,
        )
        self.assertAlmostEqual(probability, 0.5)
        self.assertAlmostEqual(ess_fraction, 0.25)

    def test_update_position_grid_likelihood_interpolates_particle_positions(self):
        centers = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        log_likelihood = np.array([0.0, 2.0, 1.0, 3.0])
        filter_ = _DummyLikelihoodFilter(np.array([[0.5, 0.5]]))

        log_marginal = update_position_grid_likelihood(
            filter_,
            log_likelihood,
            centers,
        )

        self.assertAlmostEqual(log_marginal, 1.5)

    def test_update_position_grid_likelihood_passes_discrete_grid_proposal(self):
        centers = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        log_likelihood = np.array([0.0, 2.0, 1.0, 3.0])
        filter_ = _DummyLikelihoodFilter(np.array([[0.5, 0.5]]))

        update_position_grid_likelihood(
            filter_,
            log_likelihood,
            centers,
            position_proposal_probability=1.0,
        )

        self.assertAlmostEqual(filter_.proposal_probability, 1.0)
        self.assertTrue(np.allclose(filter_.proposal_positions, centers))
        self.assertTrue(np.allclose(filter_.proposal_weights, grid_proposal_weights(log_likelihood)))

    def test_particle_position_log_posterior_accumulates_weighted_grid_masses(self):
        positions = np.asarray([[0.1, 0.0], [0.2, 0.0], [1.0, 0.0]])
        weights = np.asarray([0.25, 0.25, 0.5])
        bin_centers = np.asarray([[0.0, 0.0], [1.0, 0.0]])

        log_posterior = particle_position_log_posterior(positions, weights, bin_centers)

        self.assertTrue(np.allclose(np.exp(log_posterior), [0.5, 0.5]))


class _DummyFilter:
    def __init__(self, weights):
        self.filter_state = types.SimpleNamespace(w=np.asarray(weights, dtype=float))


class _DummyLikelihoodFilter:
    def __init__(self, positions):
        self.position_particles = np.asarray(positions, dtype=float)
        self.filter_state = types.SimpleNamespace(
            w=np.full(
                self.position_particles.shape[0],
                1.0 / self.position_particles.shape[0],
            )
        )
        self.proposal_positions = None
        self.proposal_probability = None
        self.proposal_weights = None

    def update_position_likelihood(self, likelihood, *, return_log_marginal=False):
        values = np.asarray(likelihood(self.position_particles), dtype=float)
        marginal = float(np.average(values, weights=self.filter_state.w))
        if return_log_marginal:
            return float(np.log(marginal))
        return self

    def update_position_likelihood_with_proposal(
        self,
        likelihood,
        *,
        position_proposal,
        proposal_weights=None,
        proposal_probability=1.0,
        return_log_marginal=False,
    ):
        self.proposal_positions = np.asarray(position_proposal, dtype=float)
        self.proposal_probability = float(proposal_probability)
        self.proposal_weights = None if proposal_weights is None else np.asarray(proposal_weights, dtype=float)
        return self.update_position_likelihood(likelihood, return_log_marginal=return_log_marginal)


if __name__ == "__main__":
    unittest.main()
