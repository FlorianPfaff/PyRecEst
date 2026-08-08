import copy
import operator

import numpy as np

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import eye, tile
from pyrecest.distributions import AbstractHypersphericalDistribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution import (
    HypersphericalDiracDistribution,
)

from .abstract_particle_filter import AbstractParticleFilter
from .manifold_mixins import HypersphericalFilterMixin


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


class HypersphericalParticleFilter(AbstractParticleFilter, HypersphericalFilterMixin):
    def __init__(self, n_particles, dim):
        n_particles = _validate_positive_integer(n_particles, "n_particles")
        dim = _validate_positive_integer(dim, "dim")
        HypersphericalFilterMixin.__init__(self)
        # Initialize with valid points on the sphere
        AbstractParticleFilter.__init__(
            self, HypersphericalDiracDistribution(tile(eye(dim, 1), (1, n_particles)).T)
        )

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        """Sets the filter  state to new_state if it is a type of AbstractHypersphericalDistribution."""
        if not isinstance(new_state, AbstractHypersphericalDistribution):
            raise TypeError(
                "new_state must be an instance of AbstractHypersphericalDistribution"
            )
        if not isinstance(new_state, HypersphericalDiracDistribution):
            new_state = HypersphericalDiracDistribution(
                new_state.sample(self._filter_state.d.shape[0])
            )
        self._filter_state = new_state

    def predict_identity(self, noise_distribution):
        self.predict_nonlinear(lambda x: x, noise_distribution)

    def update_identity(
        self, meas_noise, measurement, shift_instead_of_add: bool = True
    ):
        if not shift_instead_of_add:
            raise NotImplementedError()
        noise_copy = copy.deepcopy(meas_noise)
        shifted_noise = noise_copy.set_mean(measurement)
        if shifted_noise is not None:
            noise_copy = shifted_noise
        self.update_nonlinear(noise_copy.pdf)

    def update_nonlinear(self, likelihood, z=None):
        if z is None:
            self.filter_state = self.filter_state.reweigh(likelihood)
        else:
            self.filter_state = self.filter_state.reweigh(lambda x: likelihood(z, x))

    def get_estimate_mean(self):
        return self.filter_state.mean_direction()
