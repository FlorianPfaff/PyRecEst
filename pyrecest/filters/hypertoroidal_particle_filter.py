from collections.abc import Callable
from typing import Union

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    arange,
    int32,
    int64,
    linspace,
    mod,
    pi,
    random,
    sum,
    tile,
    zeros_like,
)
from pyrecest.distributions import (
    AbstractHypertoroidalDistribution,
    HypertoroidalDiracDistribution,
)

from .abstract_hypertoroidal_filter import AbstractHypertoroidalFilter
from .abstract_particle_filter import AbstractParticleFilter


class HypertoroidalParticleFilter(AbstractParticleFilter, AbstractHypertoroidalFilter):
    def __init__(
        self,
        n_particles: Union[int, int32, int64],
        dim: Union[int, int32, int64],
    ):
        if dim == 1:
            points = linspace(0.0, 2.0 * pi, num=n_particles, endpoint=False)
        else:
            points = tile(
                arange(0.0, 2.0 * pi, 2.0 * pi / n_particles), (dim, 1)
            ).T.squeeze()
        filter_state = HypertoroidalDiracDistribution(points, dim=dim)
        AbstractHypertoroidalFilter.__init__(self, filter_state)
        AbstractParticleFilter.__init__(self, filter_state)

    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution: AbstractHypertoroidalDistribution | None = None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = True,
    ):
        super().predict_nonlinear(
            f,
            noise_distribution,
            function_is_vectorized,
            shift_instead_of_add,
        )
        self.filter_state.d = mod(self.filter_state.d, 2.0 * pi)

    def predict_nonlinear_nonadditive(self, f: Callable, samples, weights):
        assert samples.shape == weights.size, "samples and weights must match in size"

        weights /= sum(weights)
        n = self.filter_state.shape[0]
        noise_ids = random.choice(arange(weights.size), size=n, p=weights)
        d = zeros_like(self.filter_state)
        for i in range(n):
            d[i, :] = f(self.filter_state[i, :], samples[noise_ids[i, :]])
        self.filter_state = d
