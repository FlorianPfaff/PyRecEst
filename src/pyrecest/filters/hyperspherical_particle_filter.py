import copy

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import eye, linalg, sum, tile
from pyrecest.distributions import AbstractHypersphericalDistribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution import (
    HypersphericalDiracDistribution,
)

from .abstract_particle_filter import AbstractParticleFilter
from .manifold_mixins import HypersphericalFilterMixin


class HypersphericalParticleFilter(AbstractParticleFilter, HypersphericalFilterMixin):
    def __init__(self, n_particles, dim):
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

    def update_identity(self, meas_noise, measurement, shift_instead_of_add: bool = True):
        if not shift_instead_of_add:
            raise NotImplementedError()
        noise_copy = copy.deepcopy(meas_noise)
        noise_copy.set_mean(measurement)
        self.update_nonlinear(noise_copy.pdf)

    def update_nonlinear(self, likelihood, z=None):
        if z is None:
            self.filter_state = self.filter_state.reweigh(likelihood)
        else:
            self.filter_state = self.filter_state.reweigh(lambda x: likelihood(z, x))

    def get_estimate_mean(self):
        vec_sum = sum(
            self.filter_state.d
            * tile(self.filter_state.w, (self.filter_state.input_dim, 1)).T,
            axis=0,
        )
        mean = vec_sum / linalg.norm(vec_sum)
        return mean
