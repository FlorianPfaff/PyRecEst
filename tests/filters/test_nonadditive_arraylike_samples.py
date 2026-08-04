import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.distributions import LinearDiracDistribution
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter


def test_predict_nonlinear_nonadditive_accepts_array_like_samples_and_weights():
    particle_filter = EuclideanParticleFilter(n_particles=3, dim=1)
    particle_filter.filter_state = LinearDiracDistribution(array([[0.0], [1.0], [2.0]]))

    particle_filter.predict_nonlinear_nonadditive(
        lambda particle, noise: particle + noise,
        samples=[[0.5], [0.5]],
        weights=[1.0, 1.0],
    )

    npt.assert_allclose(
        particle_filter.filter_state.d,
        array([[0.5], [1.5], [2.5]]),
    )
