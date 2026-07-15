import numpy.testing as npt

from pyrecest.backend import array, to_numpy
from pyrecest.distributions import LinearDiracDistribution
from pyrecest.filters import EDHParticleFlowFilter
from pyrecest.filters.daum_huang_particle_filter import gaussian_bridge_moments


def test_edh_filter_preserves_nonuniform_particle_weights():
    particles = array(
        [
            [-2.0, -1.0],
            [-1.0, 1.0],
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, 1.0],
        ]
    )
    prior = LinearDiracDistribution(
        particles, array([0.05, 0.10, 0.15, 0.25, 0.45])
    )
    expected_weights = to_numpy(prior.w).copy()
    measurement_matrix = array([[1.0, -0.5]])
    measurement = array([0.25])
    measurement_noise = array([[0.75]])
    expected_mean, expected_covariance = gaussian_bridge_moments(
        prior.mean(),
        prior.covariance(),
        measurement_matrix,
        measurement,
        measurement_noise,
        1.0,
        jitter=0.0,
    )
    filt = EDHParticleFlowFilter(
        n_particles=particles.shape[0],
        dim=particles.shape[1],
        n_steps=4,
        jitter=0.0,
    )
    filt.filter_state = prior

    filt.update_linear(measurement, measurement_matrix, measurement_noise)

    npt.assert_allclose(to_numpy(filt.filter_state.w), expected_weights)
    npt.assert_allclose(
        to_numpy(filt.filter_state.mean()), to_numpy(expected_mean), atol=1e-10
    )
    npt.assert_allclose(
        to_numpy(filt.filter_state.covariance()),
        to_numpy(expected_covariance),
        atol=1e-10,
    )
