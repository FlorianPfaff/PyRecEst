import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as FilterPyUKF
from .abstract_euclidean_filter import AbstractEuclideanFilter
from pyrecest.distributions import GaussianDistribution
from filterpy.kalman import MerweScaledSigmaPoints
import numbers
from typing import Callable

class UnscentedKalmanFilter(AbstractEuclideanFilter):
    def __init__(self, initial_state: GaussianDistribution | tuple[np.ndarray, np.ndarray],
                 dt: numbers.Real = 1,
                 fx: Callable = lambda x: x,
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
            points = MerweScaledSigmaPoints(dim_x, alpha=0.001, beta=2, kappa=0)
        self._filter_state = FilterPyUKF(dim_x=dim_x, dim_z=dim_x, dt=dt, hx=hx, fx=fx, points=points)
        self._filter_state.x = initial_state.mu
        self._filter_state.P = initial_state.C
        self._filter_state.x_prior = initial_state.mu
        self._filter_state.P_prior = initial_state.C
        # filterpy wants a prediction step as the first operation, perform a dummy one
        self.predict_identity()

    @property
    def filter_state(
        self,
    ) -> (
        GaussianDistribution | tuple[np.ndarray, np.ndarray]
    ):  # It can only return GaussianDistribution, this just serves to prevent mypy linter warnings
        return GaussianDistribution(self._filter_state.x.reshape(-1), self._filter_state.P)

    @filter_state.setter
    def filter_state(
        self, new_state: GaussianDistribution | tuple[np.ndarray, np.ndarray]
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

    def predict_nonlinear(self, fx, sys_noise_cov, **fx_args):
        self._filter_state.predict(fx=fx, Q=sys_noise_cov, **fx_args)

    def update_nonlinear(self, measurement, hx, cov_mat_meas, **hx_args):
        self._filter_state.update(z=measurement, hx=hx, R=cov_mat_meas, **hx_args)

    def predict_identity(self):
        def fx(x, y=None):
            return x

        self._filter_state.predict(fx=fx)

    def predict_linear(self, system_matrix, sys_noise_cov, sys_input=None):
        # Could convert to KF to do the step using the regular KF formulae
        if sys_input is None:
            B = None
        else:
            B = np.eye(sys_input.shape[0])
        
        if B is None:
            fx = lambda x : system_matrix @ x
        else:
            fx = lambda x : system_matrix @ x + B @ sys_input
        self._filter_state.predict(fx=fx, Q=sys_noise_cov)

    def update_identity(self, meas, meas_cov):
        # Could convert to KF to do the step using the regular KF formulae
        
        hx = lambda x : x
        self._filter_state.update(z=np.atleast_1d(meas), R=meas_cov, hx=hx)

    def update_linear(self, measurement, measurement_matrix, cov_mat_meas):
        # Could convert to KF to do the step using the regular KF formulae
        
        hx = lambda x : measurement_matrix @ x
        self._filter_state.update(z=measurement, R=cov_mat_meas, hx=hx)

    def get_estimate(self):
        return GaussianDistribution(self._filter_state.x, self._filter_state.P)
