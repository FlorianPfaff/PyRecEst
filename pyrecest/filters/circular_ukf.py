from .abstract_circular_filter import AbstractCircularFilter
from pyrecest.distributions import GaussianDistribution
import numpy as np
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter
from pyrecest.distributions import GaussianDistribution, AbstractCircularDistribution
import copy
from beartype import beartype

class CircularUKF(AbstractCircularFilter):
    def __init__(self, initial_prior=None):
        if initial_prior is None:
            initial_prior = GaussianDistribution(0, 1)
        self.filter_state = initial_prior
        
    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    @beartype
    def filter_state(self, new_state: AbstractCircularDistribution):
        if not isinstance(new_state, GaussianDistribution):
            new_state = new_state.to_gaussian()
        assert np.size(new_state.mu) == 1
        self._filter_state = copy.deepcopy(new_state)

    @beartype
    def predict_identity(self, gauss_sys: AbstractCircularDistribution):
        if not isinstance(gauss_sys, GaussianDistribution):
            gauss_sys = gauss_sys.to_gaussian()
        mu = np.mod(self.state.mu + gauss_sys.mu, 2 * np.pi)
        C = self.state.C + gauss_sys.C
        self.filter_state = GaussianDistribution(mu, C)

    @beartype
    def update_identity(self, gauss_meas, z: AbstractCircularDistribution):
        assert np.isscalar(z)
        if not isinstance(gauss_meas, GaussianDistribution):
            gauss_meas = gauss_meas.to_gaussian()
        z = np.mod(z - gauss_meas.mu, 2 * np.pi)

        if abs(self.filter_state.mu - z) > np.pi:
            z = z + 2 * np.pi * np.sign(self.filter_state.mu - z)

        # K is scalar so no matrix multiplication required
        K = self.filter_state.C / (self.filter_state.C + gauss_meas.C)
        mu = self.filter_state.mu + K * (z - self.state.mu)
        C = (1 - K) * self.filter_state.C  # K is scalar so I = 1

        mu = np.mod(mu, 2 * np.pi)
        self.filter_state = GaussianDistribution(mu, C)
