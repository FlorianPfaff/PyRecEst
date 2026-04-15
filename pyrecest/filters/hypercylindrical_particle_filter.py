from collections.abc import Callable
from typing import Union

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import concatenate, int32, int64, mod, ones, pi, zeros
from pyrecest.distributions.cart_prod.hypercylindrical_dirac_distribution import (
    HypercylindricalDiracDistribution,
)

from .abstract_particle_filter import AbstractParticleFilter
from .manifold_mixins import HypercylindricalFilterMixin


class HypercylindricalParticleFilter(
    AbstractParticleFilter, HypercylindricalFilterMixin
):
    def __init__(
        self,
        n_particles: Union[int, int32, int64],
        bound_dim: Union[int, int32, int64],
        lin_dim: Union[int, int32, int64],
    ):
        d = zeros((n_particles, bound_dim + lin_dim))
        w = ones(n_particles) / n_particles
        filter_state = HypercylindricalDiracDistribution(bound_dim, d, w)
        HypercylindricalFilterMixin.__init__(self)
        AbstractParticleFilter.__init__(self, filter_state)

    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution=None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = True,
    ):
        super().predict_nonlinear(
            f, noise_distribution, function_is_vectorized, shift_instead_of_add
        )
        # Wrap periodic dimensions to [0, 2*pi)
        bound_dim = self.filter_state.bound_dim
        wrapped_periodic = mod(self.filter_state.d[:, :bound_dim], 2.0 * pi)
        linear_part = self.filter_state.d[:, bound_dim:]
        self.filter_state.d = concatenate([wrapped_periodic, linear_part], axis=1)
