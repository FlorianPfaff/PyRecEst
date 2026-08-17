import numpy.testing as npt
import pytest
from pyrecest.backend import array
from pyrecest.distributions import LinearDiracDistribution
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter


def test_predict_nonlinear_nonadditive_rejects_changed_particle_shape():
    particle_filter = EuclideanParticleFilter(n_particles=3, dim=1)
    initial_particles = array([[0.0], [1.0], [2.0]])
    particle_filter.filter_state = LinearDiracDistribution(initial_particles)

    with pytest.raises(
        ValueError,
        match="Nonadditive transition returned particles with shape",
    ):
        particle_filter.predict_nonlinear_nonadditive(
            lambda particle, noise: particle + noise[0] + array([0.0, 1.0]),
            samples=[[0.0]],
            weights=[1.0],
        )

    npt.assert_allclose(particle_filter.filter_state.d, initial_particles)
    npt.assert_allclose(
        particle_filter.filter_state.w,
        array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    )
