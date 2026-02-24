from bayesian_filters.kalman import UnscentedKalmanFilter as BayesianFiltersUKF
from bayesian_filters.kalman import MerweScaledSigmaPoints

from .manifold_mixins import EuclideanFilterMixin
from pyrecest.distributions import GaussianDistribution

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import atleast_1d
import numbers
from typing import Callable

class UnscentedKalmanFilter(EuclideanFilterMixin):
    def __init__(self, initial_state: GaussianDistribution | tuple,
                 dt: numbers.Real = 1,
                 fx: Callable = lambda x, dt: x,
                 hx: Callable = lambda x: x,
                 points=None,
    ):
        if isinstance(initial_state, GaussianDistribution):
            dim_x = initial_state.dim
        elif isinstance(initial_state, tuple) and len(initial_state) == 2:
            dim_x = len(initial_state[0])
        else:
            raise ValueError(
                "initial_state must be a GaussianDistribution or a tuple of (mean, covariance)"
            )
            
        if points is None:
            # Standard settings for Gaussian approximations
            points = MerweScaledSigmaPoints(dim_x, alpha=0.001, beta=2, kappa=0)
            
        # Initialize bayesian_filters UKF
        # Note: We initialize dim_z to dim_x as a default, but this can be 
        # overridden dynamically in update() by providing R and hx
        self._filter_state = BayesianFiltersUKF(dim_x=dim_x, dim_z=dim_x, dt=dt, hx=hx, fx=fx, points=points)
        
        # Set initial state
        if isinstance(initial_state, GaussianDistribution):
            self._filter_state.x = initial_state.mu
            self._filter_state.P = initial_state.C
        else:
            self._filter_state.x = initial_state[0]
            self._filter_state.P = initial_state[1]

        self._filter_state.x_prior = self._filter_state.x.copy()
        self._filter_state.P_prior = self._filter_state.P.copy()

    @property
    def filter_state(
        self,
    ) -> (
        GaussianDistribution | tuple
    ):  # It can only return GaussianDistribution, this just serves to prevent mypy linter warnings
        return GaussianDistribution(self._filter_state.x, self._filter_state.P)

    @filter_state.setter
    def filter_state(
        self, new_state: GaussianDistribution | tuple
    ):
        """
        Set the filter state.

        :param new_state: Provide GaussianDistribution or mean and covariance as state.
        """
        if isinstance(new_state, GaussianDistribution):
            self._filter_state.x = new_state.mu
            self._filter_state.P = new_state.C
        elif isinstance(new_state, tuple) and len(new_state) == 2:
            self._filter_state.x = new_state[0]
            self._filter_state.P = new_state[1]
        else:
            raise ValueError(
                "new_state must be a GaussianDistribution or a tuple of (mean, covariance)"
            )

    def predict_nonlinear(self, fx, sys_noise_cov, dt=None, **fx_args):
        """
        :param fx: Function with signature fx(x, dt, **fx_args)
        :param sys_noise_cov: Process noise matrix Q
        """
        # FIX: FilterPy UKF uses member variable Q, not an argument to predict()
        self._filter_state.Q = sys_noise_cov
        self._filter_state.predict(dt=dt, fx=fx, **fx_args)

    def update_nonlinear(self, measurement, hx, cov_mat_meas, **hx_args):
        # Update allows R to be passed as argument
        self._filter_state.update(z=measurement, hx=hx, R=cov_mat_meas, **hx_args)

    def predict_identity(self, sys_noise_cov, dt=None):
        self._filter_state.Q = sys_noise_cov
        def fx(x, _):
            return x

        self._filter_state.predict(dt=dt, fx=fx)

    def predict_linear(self, system_matrix, sys_noise_cov, sys_input=None, dt=None):
        self._filter_state.Q = sys_noise_cov
        
        if sys_input is None:
            def fx(x, _):
                # F * x
                return system_matrix @ x
        else:
            def fx(x, _):
                # F * x + B * u
                # sys_input should already be B*u
                return system_matrix @ x + sys_input

        self._filter_state.predict(dt=dt, fx=fx)

    def update_identity(self, meas, meas_cov):
        def hx(x):
            # h(x) = x
            return x
        self._filter_state.update(z=atleast_1d(meas), R=meas_cov, hx=hx)

    def update_linear(self, measurement, measurement_matrix, cov_mat_meas):
        def hx(x):
            # h(x) = H * x
            return measurement_matrix @ x
        
        self._filter_state.update(z=measurement, R=cov_mat_meas, hx=hx)

    def get_estimate(self):
        return GaussianDistribution(self._filter_state.x, self._filter_state.P)