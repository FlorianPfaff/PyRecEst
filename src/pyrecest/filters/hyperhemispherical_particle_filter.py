import operator

import numpy as np

from pyrecest.backend import empty
from pyrecest.distributions.hypersphere_subset.abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_dirac_distribution import (
    HyperhemisphericalDiracDistribution,
)

from .abstract_particle_filter import AbstractParticleFilter
from .manifold_mixins import HyperhemisphericalFilterMixin


def _is_boolean_scalar(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    dtype = getattr(value, "dtype", None)
    if getattr(dtype, "kind", None) == "b":
        return True
    return str(dtype).lower() in {"bool", "bool_", "torch.bool"}


def _validate_positive_integer(value, name: str) -> int:
    message = f"{name} must be a positive integer."
    if np.ma.is_masked(value) or _is_boolean_scalar(value):
        raise ValueError(message)
    try:
        value = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if value <= 0:
        raise ValueError(message)
    return value


class HyperhemisphericalParticleFilter(
    AbstractParticleFilter, HyperhemisphericalFilterMixin
):
    def __init__(self, n_particles: int, dim: int) -> None:
        """
        Constructor

        Parameters:
        n_particles (int > 0): Number of particles to use
        dim (int > 0): Dimension
        """
        n_particles = _validate_positive_integer(n_particles, "n_particles")
        dim = _validate_positive_integer(dim, "dim")
        initial_filter_state = HyperhemisphericalDiracDistribution(
            empty((n_particles, dim + 1))
        )
        HyperhemisphericalFilterMixin.__init__(self)
        AbstractParticleFilter.__init__(self, initial_filter_state=initial_filter_state)

    def set_state(self, new_state):
        """
        Sets the current system state

        Parameters:
        dist_ (HyperhemisphericalDiracDistribution): New state
        """
        if not isinstance(new_state, AbstractHyperhemisphericalDistribution):
            raise TypeError(
                "new_state must be an AbstractHyperhemisphericalDistribution."
            )
        if not isinstance(new_state, HyperhemisphericalDiracDistribution):
            new_state = HyperhemisphericalDiracDistribution(
                new_state.sample(self.filter_state.d.shape[0])
            )
        self.filter_state = new_state
