"""Regression tests for numerically stable box-particle corrections."""

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.nonperiodic.linear_box_particle_distribution import (
    LinearBoxParticleDistribution,
)
from pyrecest.filters.euclidean_box_particle_filter import EuclideanBoxParticleFilter


def test_contracted_update_preserves_posterior_when_products_underflow():
    particle_filter = EuclideanBoxParticleFilter(2, 1)
    particle_filter.filter_state = LinearBoxParticleDistribution(
        array([[0.0], [2.0]]),
        array([[1.0], [3.0]]),
        array([1e-200, 1.0]),
    )
    particle_filter.set_resampling_criterion(lambda _state: False)

    particle_filter.update_contracted(
        lambda lower, upper: (lower, upper),
        likelihood=lambda _centers: array([1e-200, 0.0]),
    )

    npt.assert_allclose(particle_filter.filter_state.w, array([1.0, 0.0]))
