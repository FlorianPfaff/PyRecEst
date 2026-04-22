"""
Unscented Kalman filter for distributions on the unit hypersphere.

References:
    Gerhard Kurz, Igor Gilitschenski, Uwe D. Hanebeck,
    Recursive Bayesian Filtering in Circular State Spaces,
    arXiv preprint: Systems and Control (cs.SY), January 2015.

Ported from the MATLAB libDirectional library:
    https://github.com/libDirectional/libDirectional/blob/master/lib/filters/HypersphericalUKF.m
"""

# pylint: disable=no-name-in-module,no-member,duplicate-code
from typing import Callable

import pyrecest.backend
from pyrecest.backend import (  # pylint: disable=redefined-builtin
    all,
    array,
    asarray,
    empty,
    expand_dims,
    eye,
    float64,
    linalg,
    reshape,
    zeros,
)
from pyrecest.distributions import GaussianDistribution

from ._ukf import MerweScaledSigmaPoints
from ._ukf import UnscentedKalmanFilter as BayesianFiltersUKF
from ._ukf import _UKFModel
from .abstract_filter import AbstractFilter
from .manifold_mixins import HypersphericalFilterMixin


class HypersphericalUKF(AbstractFilter, HypersphericalFilterMixin):
    """
    Unscented Kalman filter on the unit hypersphere S^(d-1).

    The state is represented as a d-dimensional :class:`GaussianDistribution`
    whose mean is kept on the unit hypersphere via normalization after each
    prediction/update step.

    Parameters
    ----------
    dim:
        Embedding-space dimension (e.g. ``2`` for S^1, ``3`` for S^2).
    alpha, beta, kappa:
        Sigma-point spread parameters for :class:`MerweScaledSigmaPoints`.
    """

    def __init__(
        self,
        dim: int = 2,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        assert pyrecest.backend.__backend_name__ not in (
            "jax",
        ), "HypersphericalUKF is not supported on the JAX backend"
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        mu0 = zeros(dim)
        mu0[0] = 1.0
        initial_state = GaussianDistribution(mu0, eye(dim))
        HypersphericalFilterMixin.__init__(self)
        AbstractFilter.__init__(self, initial_state)

    # ------------------------------------------------------------------
    # filter_state property
    # ------------------------------------------------------------------

    @property
    def filter_state(self) -> GaussianDistribution:
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if not isinstance(new_state, GaussianDistribution):
            new_state = GaussianDistribution.from_distribution(new_state)
        self._filter_state = new_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_sigma_points(self, dim_x: int) -> MerweScaledSigmaPoints:
        return MerweScaledSigmaPoints(
            n=dim_x, alpha=self._alpha, beta=self._beta, kappa=self._kappa
        )

    def _build_ukf(
        self, dim_x: int, dim_z: int, fx: Callable, hx: Callable
    ) -> BayesianFiltersUKF:
        """Build a BayesianFiltersUKF initialized from the current filter state."""
        points = self._make_sigma_points(dim_x)
        ukf = BayesianFiltersUKF(
            _UKFModel(dim_x=dim_x, dim_z=dim_z, dt=1.0, hx=hx, fx=fx, points=points)
        )
        ukf.x = reshape(asarray(self._filter_state.mu, dtype=float64), (-1,))
        ukf.P = asarray(self._filter_state.C, dtype=float64)
        return ukf

    @staticmethod
    def _normalize(x):
        n = linalg.norm(x)
        if n == 0.0:
            raise ValueError("Mean is zero; normalization failed.")
        return x / n

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_nonlinear(self, f: Callable, gauss_sys: GaussianDistribution):
        """
        Predict assuming a nonlinear system model:
            x(k+1) = normalize(f(normalize(x(k)))) + w(k)

        Parameters
        ----------
        f:
            Function from S^(d-1) to S^(d-1).
        gauss_sys:
            Distribution of additive system noise (mean is ignored).
        """
        assert pyrecest.backend.__backend_name__ not in (
            "pytorch",
            "jax",
        ), "Not supported on this backend"

        if not isinstance(gauss_sys, GaussianDistribution):
            gauss_sys = GaussianDistribution.from_distribution(gauss_sys)

        Q = asarray(gauss_sys.C, dtype=float64)
        dim_x = self._filter_state.dim

        def _fx(x, _dt):
            x_unit = self._normalize(x)
            y = reshape(asarray(f(array(x_unit)), dtype=float64), (-1,))
            return self._normalize(y)

        ukf = self._build_ukf(dim_x, dim_x, _fx, lambda x: x)
        ukf.Q = Q
        ukf.predict()

        mu = self._normalize(reshape(ukf.x, (-1,)))
        self._filter_state = GaussianDistribution(
            array(mu), array(ukf.P), check_validity=False
        )

    def predict_nonlinear_arbitrary_noise(  # pylint: disable=too-many-locals
        self,
        f: Callable,
        noise_samples,
        noise_weights,
    ):
        """
        Predict assuming nonlinear system model with arbitrary noise:
            x(k+1) = f(x(k), v_k)

        Parameters
        ----------
        f:
            Function ``f(x, v) -> x_new`` where x is a unit vector in R^d.
        noise_samples:
            Array of shape ``(noise_dim, n_noise)`` with noise samples (columns).
        noise_weights:
            Array of length ``n_noise`` with positive weights.
        """
        assert pyrecest.backend.__backend_name__ not in (
            "pytorch",
            "jax",
        ), "Not supported on this backend"

        noise_samples = asarray(noise_samples, dtype=float64)
        noise_weights = reshape(asarray(noise_weights, dtype=float64), (-1,))
        assert noise_samples.shape[1] == noise_weights.shape[0]
        assert all(noise_weights > 0)
        noise_weights = noise_weights / noise_weights.sum()

        mu = reshape(asarray(self._filter_state.mu, dtype=float64), (-1,))
        C = asarray(self._filter_state.C, dtype=float64)
        dim_x = mu.shape[0]

        points = self._make_sigma_points(dim_x)
        sigmas = points.sigma_points(mu, C)  # shape: (2*dim_x+1, dim_x)
        state_weights = asarray(points.Wm, dtype=float64)

        n_sigmas = sigmas.shape[0]
        n_noise = noise_samples.shape[1]

        new_samples = empty((dim_x, n_sigmas * n_noise))
        new_weights = empty((n_sigmas * n_noise,))
        k = 0
        for i in range(n_sigmas):
            for j in range(n_noise):
                x_new = reshape(
                    asarray(
                        f(array(sigmas[i]), array(noise_samples[:, j])),
                        dtype=float64,
                    ),
                    (-1,),
                )
                new_samples[:, k] = x_new
                new_weights[k] = noise_weights[j] * state_weights[i]
                k += 1

        new_weights = new_weights / new_weights.sum()

        # Weighted mean and covariance
        predicted_mean = (new_samples * expand_dims(new_weights, 0)).sum(axis=1)
        diff = new_samples - expand_dims(predicted_mean, -1)
        predicted_cov = (diff * expand_dims(new_weights, 0)) @ diff.T

        predicted_mean = self._normalize(predicted_mean)
        self._filter_state = GaussianDistribution(
            array(predicted_mean), array(predicted_cov), check_validity=False
        )

    def predict_identity(self, gauss_sys: GaussianDistribution):
        """
        Predict with identity system model:
            x(k+1) = x(k) + w(k)

        Parameters
        ----------
        gauss_sys:
            Distribution of additive system noise (mean is ignored).
        """
        self.predict_nonlinear(lambda x: x, gauss_sys)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_nonlinear(self, f: Callable, gauss_meas: GaussianDistribution, z):
        """
        Update assuming a nonlinear measurement model:
            z(k) = f(normalize(x(k))) + v_k

        Parameters
        ----------
        f:
            Measurement function from S^(d-1) to R^m.
        gauss_meas:
            Distribution of additive measurement noise (mean is ignored).
        z:
            Measurement vector of shape (m,) or scalar.
        """
        assert pyrecest.backend.__backend_name__ not in (
            "pytorch",
            "jax",
        ), "Not supported on this backend"

        if not isinstance(gauss_meas, GaussianDistribution):
            gauss_meas = GaussianDistribution.from_distribution(gauss_meas)

        z_arr = reshape(asarray(z, dtype=float64), (-1,))
        dim_z = z_arr.shape[0]
        dim_x = self._filter_state.dim
        R = asarray(gauss_meas.C, dtype=float64)

        def _hx(x):
            x_unit = self._normalize(x)
            return reshape(asarray(f(array(x_unit)), dtype=float64), (-1,))

        ukf = self._build_ukf(dim_x, dim_z, lambda x, _dt: x, _hx)
        ukf.Q = zeros((dim_x, dim_x))
        ukf.R = R
        ukf.predict()
        ukf.update(z_arr)

        mu = self._normalize(reshape(ukf.x, (-1,)))
        self._filter_state = GaussianDistribution(
            array(mu), array(ukf.P), check_validity=False
        )

    def update_identity(self, gauss_meas: GaussianDistribution, z):
        """
        Update with identity measurement model:
            z(k) = x(k) + v_k

        Parameters
        ----------
        gauss_meas:
            Distribution of additive measurement noise (mean is ignored).
        z:
            Measurement vector on S^(d-1).
        """
        self.update_nonlinear(lambda x: x, gauss_meas, z)

    # ------------------------------------------------------------------
    # Point estimate
    # ------------------------------------------------------------------

    def get_point_estimate(self):
        """Return the mean of the current state estimate (unit vector)."""
        return self._filter_state.mu
