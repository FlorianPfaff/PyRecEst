# pylint: disable=no-name-in-module,no-member,duplicate-code
from typing import Callable

import pyrecest.backend
from pyrecest.backend import asarray, atleast_1d, float64, reshape, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.models import AdditiveNoiseMeasurementModel, AdditiveNoiseTransitionModel
from pyrecest.sampling.sigma_points import MerweScaledSigmaPoints

from ._ukf import UnscentedKalmanFilter as BayesianFiltersUKF
from ._ukf import _UKFModel
from .abstract_filter import AbstractFilter
from .manifold_mixins import EuclideanFilterMixin


class UnscentedKalmanFilter(AbstractFilter, EuclideanFilterMixin):
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        initial_state: "GaussianDistribution | tuple",
        dt: float = 1.0,
        fx: Callable = lambda x, dt: x,
        hx: Callable = lambda x: x,
        points=None,
        dim_z: int | None = None,
    ):
        if isinstance(initial_state, GaussianDistribution):
            dim_x = initial_state.dim
            initial_mean = initial_state.mu
        elif isinstance(initial_state, tuple) and len(initial_state) == 2:
            dim_x = len(initial_state[0])
            initial_mean = initial_state[0]
        else:
            raise ValueError(
                "initial_state must be a GaussianDistribution or a tuple of (mean, covariance)"
            )
        dim_z = (
            reshape(asarray(hx(initial_mean), dtype=float64), (-1,)).shape[0]
            if dim_z is None
            else dim_z
        )

        if points is None:
            # Standard settings for Gaussian approximations
            points = MerweScaledSigmaPoints(dim_x, alpha=0.001, beta=2, kappa=0)

        EuclideanFilterMixin.__init__(self)
        bfukf = BayesianFiltersUKF(
            _UKFModel(dim_x=dim_x, dim_z=dim_z, dt=dt, hx=hx, fx=fx, points=points)
        )
        AbstractFilter.__init__(self, bfukf)

        # Set initial state
        if isinstance(initial_state, GaussianDistribution):
            self._filter_state.x = initial_state.mu
            self._filter_state.P = initial_state.C
        else:
            self._filter_state.x = initial_state[0]
            self._filter_state.P = initial_state[1]
        # Track whether predict() has been called before update()
        self._predicted = False

    @property
    def filter_state(
        self,
    ) -> GaussianDistribution:
        return GaussianDistribution(self._filter_state.x, self._filter_state.P)

    @filter_state.setter
    def filter_state(self, new_state: "GaussianDistribution | tuple"):
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

    def _ensure_predicted(self):
        """Run a zero-noise predict to populate sigma points if needed."""
        if not self._predicted:
            dim_x = self._filter_state.x.shape[0]
            self._filter_state.Q = zeros((dim_x, dim_x))

            def _identity_fx(x, _dt):
                return x

            self._filter_state.predict(fx=_identity_fx)
            self._predicted = True

    def predict_nonlinear(self, fx, sys_noise_cov, dt=None, **fx_args):
        """
        :param fx: Function with signature fx(x, dt, **fx_args)
        :param sys_noise_cov: Process noise matrix Q
        """
        assert pyrecest.backend.__backend_name__ not in (
            "pytorch",
            "jax",
        ), "Not supported on this backend"
        self._filter_state.Q = sys_noise_cov
        self._filter_state.predict(dt=dt, fx=fx, **fx_args)
        self._predicted = True

    def update_nonlinear(self, measurement, hx, cov_mat_meas, **hx_args):
        assert pyrecest.backend.__backend_name__ not in (
            "pytorch",
            "jax",
        ), "Not supported on this backend"
        self._ensure_predicted()
        self._filter_state.update(z=measurement, hx=hx, R=cov_mat_meas, **hx_args)
        self._predicted = False

    def predict_model(
        self,
        model: AdditiveNoiseTransitionModel,
        dt=None,
        **fx_args,
    ):
        """Run a prediction step using an additive-noise transition model.

        This is an adapter around :meth:`predict_nonlinear`; it does not change
        the UKF algorithm or deprecate the existing function/covariance API.

        Parameters
        ----------
        model:
            Reusable transition model containing the deterministic transition
            function and additive process-noise covariance.
        dt:
            Optional time step overriding ``model.dt``.
        fx_args:
            Optional transition-function keyword arguments overriding the
            model's default ``function_args`` for this prediction.
        """
        return self.predict_nonlinear(
            model.evaluate,
            model.noise_covariance,
            dt=model.dt if dt is None else dt,
            **fx_args,
        )

    def update_model(
        self,
        model: AdditiveNoiseMeasurementModel,
        measurement,
        **hx_args,
    ):
        """Run an update step using an additive-noise measurement model.

        This is an adapter around :meth:`update_nonlinear`; it preserves the
        existing direct nonlinear update API while allowing model-object reuse.
        """
        return self.update_nonlinear(
            measurement,
            model.evaluate,
            model.noise_covariance,
            **hx_args,
        )

    def predict_identity(self, sys_noise_cov, dt=None):
        assert pyrecest.backend.__backend_name__ not in (
            "pytorch",
            "jax",
        ), "Not supported on this backend"
        self._filter_state.Q = sys_noise_cov

        def fx(x, _):
            return x

        self._filter_state.predict(dt=dt, fx=fx)
        self._predicted = True

    def predict_linear(self, system_matrix, sys_noise_cov, sys_input=None, dt=None):
        assert pyrecest.backend.__backend_name__ not in (
            "pytorch",
            "jax",
        ), "Not supported on this backend"
        self._filter_state.Q = sys_noise_cov

        if sys_input is None:

            def fx(x, _):
                return system_matrix @ x

        else:

            def fx(x, _):
                return system_matrix @ x + sys_input

        self._filter_state.predict(dt=dt, fx=fx)
        self._predicted = True

    def update_identity(self, meas, meas_cov):
        assert pyrecest.backend.__backend_name__ not in (
            "pytorch",
            "jax",
        ), "Not supported on this backend"
        self._ensure_predicted()

        def hx(x):
            return x

        self._filter_state.update(z=atleast_1d(meas), R=meas_cov, hx=hx)
        self._predicted = False

    def update_linear(self, measurement, measurement_matrix, cov_mat_meas):
        assert pyrecest.backend.__backend_name__ not in (
            "pytorch",
            "jax",
        ), "Not supported on this backend"
        self._ensure_predicted()

        def hx(x):
            return measurement_matrix @ x

        self._filter_state.update(z=measurement, R=cov_mat_meas, hx=hx)
        self._predicted = False

    def get_point_estimate(self):
        return self._filter_state.x
