import numpy as np

from pyrecest.filters import particle_position_log_posterior


def test_particle_position_log_posterior_preserves_finite_log_zero():
    positions = np.asarray([[0.0], [1.0]])
    weights = np.asarray([0.25, 0.75])
    bin_centers = np.asarray([[0.0], [1.0], [2.0]])

    log_posterior = particle_position_log_posterior(
        positions,
        weights,
        bin_centers,
        log_zero=-10.0,
    )
    alternate = particle_position_log_posterior(
        positions,
        weights,
        bin_centers,
        log_zero=-100.0,
    )

    np.testing.assert_allclose(
        log_posterior[:2],
        np.log(weights),
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        alternate[:2],
        log_posterior[:2],
        rtol=0.0,
        atol=1e-14,
    )
    assert log_posterior[2] == -10.0
    assert alternate[2] == -100.0
