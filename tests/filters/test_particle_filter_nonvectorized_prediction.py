import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter


def test_nonvectorized_prediction_calls_transition_per_particle():
    particles = array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    particle_filter = EuclideanParticleFilter(n_particles=4, dim=2)
    particle_filter.filter_state = LinearDiracDistribution(particles)
    observed_shapes = []

    def transition(particle):
        observed_shapes.append(tuple(particle.shape))
        x_coord, y_coord = particle
        return array([x_coord + y_coord, x_coord - y_coord])

    particle_filter.predict_nonlinear(
        transition,
        noise_distribution=None,
        function_is_vectorized=False,
    )

    npt.assert_allclose(
        particle_filter.filter_state.d,
        array(
            [
                [3.0, -1.0],
                [7.0, -1.0],
                [11.0, -1.0],
                [15.0, -1.0],
            ]
        ),
    )
    assert observed_shapes == [(2,), (2,), (2,), (2,)]
