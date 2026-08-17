import numpy as np
from pyrecest.filters.replay_grid_likelihood import (
    particle_position_log_posterior,
    replay_grid_log_likelihood_values,
)


def test_ckdtree_overflow_falls_back_to_stable_nearest_bin():
    bin_centers = np.asarray([[0.0, 0.0], [2.0e155, 0.0]])
    positions = np.asarray([[1.1e155, 0.0]])
    log_likelihood = np.asarray([3.0, 7.0])

    likelihood_values = replay_grid_log_likelihood_values(
        positions,
        log_likelihood,
        bin_centers,
        interpolation="nearest",
    )
    log_posterior = particle_position_log_posterior(
        positions,
        np.asarray([1.0]),
        bin_centers,
    )

    np.testing.assert_allclose(likelihood_values, [7.0])
    np.testing.assert_allclose(np.exp(log_posterior), [0.0, 1.0])
