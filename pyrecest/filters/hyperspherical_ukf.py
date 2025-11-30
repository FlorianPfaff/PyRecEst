import numpy as np
from pyrecest.distributions import GaussianDistribution
from .unscented_kalman_filter import UnscentedKalmanFilter
from .abstract_hyperspherical_filter import AbstractHyperSphericalFilter


class HypersphericalUKF(AbstractHyperSphericalFilter):
    def __init__(self):
        # Constructor
        self.state = GaussianDistribution(1, 1)
        self.ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1, dt=1)

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        # Sets the current system state
        #
        # Parameters:
        #   new_state (GaussianDistribution)
        #       new state (1D Gaussian)
        if not isinstance(new_state, GaussianDistribution):
            new_state = new_state.toGaussian()
        self._filter_state = new_state

    def predictNonlinear(self, f, gaussSys):
        # Predicts assuming a nonlinear system model, i.e.,
        # x(k+1) = f(x(k)) + w(k)
        # where w(k) is additive noise given by gaussSys.
        #
        # Parameters:
        #   f (function handle)
        #       function from R^(d-1) to R^(d-1)
        #   gaussSys (GaussianDistribution)
        #       distribution of additive noise (warning: mean is
        #       ignored)
        assert callable(f)
        if not isinstance(gaussSys, GaussianDistribution):
            gaussSys = gaussSys.toGaussian()

        def g(x):
            x = x / np.linalg.norm(x)
            y = f(x)
            y = y / np.linalg.norm(y)
            return y

        self.ukf.x = self.state.mu
        self.ukf.P = self.state.C
        self.ukf.predict(fx=g, Q=gaussSys.C)
        self.state.mu, self.state.C = self.ukf.x, self.ukf.P

        # normalize mean
        if np.linalg.norm(self.state.mu) != 0:
            self.state.mu = self.state.mu / np.linalg.norm(self.state.mu)
        else:
            raise ValueError('mu was 0')
        
    def getEstimateMean(self):
        return self.state.mu