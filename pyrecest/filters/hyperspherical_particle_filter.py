from .abstract_hyperspherical_filter import AbstractHypersphericalFilter
from pyrecest.distributions import VonMisesDistribution, AbstractHypersphericalDistribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution import HypersphericalDiracDistribution
import numpy as np
import warnings
from pyrecest.backend import tile, eye, sum, linalg

class HypersphericalParticleFilter(AbstractHypersphericalFilter):
    def __init__(self, n_particles, dim):
        self.wd = HypersphericalDiracDistribution(tile(eye(dim, 1), (1, n_particles)))

    @property
    def filter_state(self):
        return self._filter_state
    
    @filter_state.setter
    def filter_state(self, new_state):
        """Sets the filter  state to new_state if it is a type of AbstractHypersphericalDistribution."""
        if not isinstance(new_state, AbstractHypersphericalDistribution):
            raise TypeError("new_state must be an instance of AbstractHypersphericalDistribution")
        if not isinstance(new_state, HypersphericalDiracDistribution):
            new_state = HypersphericalDiracDistribution(new_state.sample(np.size(self._filter_state.d)))
        self._filter_state = new_state

    def predict_identity(self, noise_distribution):
        self.predict_nonlinear(lambda x: x, noise_distribution)

    def update_identity(self, noise_distribution, z):
        assert isinstance(noise_distribution, VonMisesDistribution), "Currently, only VMF distributed noise terms are supported."
        if z is not None:
            noise_distribution.mu = z
            warnings.warn("Warning: update_identity: mu of noise_distribution is replaced by measurement...")

    def update_nonlinear(self, likelihood, z=None):
        if z is None:
            self.wd = self.wd.reweigh(likelihood)
        else:
            self.wd = self.wd.reweigh(lambda x: likelihood(z, x))

    def get_estimate_mean(self):
        vec_sum = sum(self.wd.d * tile(self.wd.w, (self.dim, 1)), axis=1)
        mean = vec_sum / linalg.norm(vec_sum)
        return mean