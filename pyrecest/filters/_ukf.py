"""
Implementations of Merwe-scaled and Julier sigma-point generators and an
Unscented Kalman Filter (UKF), replacing the former dependency on the
``bayesian_filters`` package.
"""

from collections import namedtuple
from copy import deepcopy

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    asarray,
    copy,
    einsum,
    empty,
    empty_like,
    expand_dims,
    eye,
    float64,
    full,
    linalg,
    reshape,
    transpose,
    zeros,
)

# ---------------------------------------------------------------------------
# Model configuration container
# ---------------------------------------------------------------------------

_UKFModel = namedtuple("_UKFModel", ["dim_x", "dim_z", "dt", "hx", "fx", "points"])

# ---------------------------------------------------------------------------
# Sigma-point generators
# ---------------------------------------------------------------------------


class MerweScaledSigmaPoints:
    """Merwe scaled sigma points (van der Merwe, 2004).

    Parameters
    ----------
    n:
        State dimension.
    alpha:
        Spread of sigma points around the mean (typically 1e-3).
    beta:
        Prior knowledge of the distribution (2 is optimal for Gaussians).
    kappa:
        Secondary scaling parameter (typically 0).
    """

    def __init__(self, n: int, alpha: float, beta: float, kappa: float):
        assert pyrecest.backend.__backend_name__ not in (
            "jax",
        ), "MerweScaledSigmaPoints is not supported on the JAX backend"
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._compute_weights()

    # ------------------------------------------------------------------
    def _compute_weights(self):
        n = self.n
        lam = self.alpha**2 * (n + self.kappa) - n

        self.Wm = full(2 * n + 1, 0.5 / (n + lam))
        self.Wm[0] = lam / (n + lam)

        self.Wc = copy(self.Wm)
        self.Wc[0] = lam / (n + lam) + (1.0 - self.alpha**2 + self.beta)

    # ------------------------------------------------------------------
    def sigma_points(self, x, P):
        """Return ``(2n+1, n)`` sigma-point matrix.

        Parameters
        ----------
        x:
            State mean, shape ``(n,)``.
        P:
            State covariance, shape ``(n, n)``.
        """
        n = self.n
        lam = self.alpha**2 * (n + self.kappa) - n

        x = reshape(asarray(x, dtype=float64), (-1,))
        P = asarray(P, dtype=float64)

        U = linalg.cholesky((n + lam) * P)  # lower-triangular

        sigmas = empty((2 * n + 1, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1] = x + U[:, i]
            sigmas[n + i + 1] = x - U[:, i]
        return sigmas


class JulierSigmaPoints:
    """Julier sigma points (Julier & Uhlmann, 1997).

    Parameters
    ----------
    n:
        State dimension.
    kappa:
        Scaling parameter (``n + kappa`` should be non-zero).
    """

    def __init__(self, n: int, kappa: float = 0.0):
        assert pyrecest.backend.__backend_name__ not in (
            "jax",
        ), "JulierSigmaPoints is not supported on the JAX backend"
        self.n = n
        self.kappa = kappa
        self._compute_weights()

    # ------------------------------------------------------------------
    def _compute_weights(self):
        n = self.n
        k = n + self.kappa

        self.Wm = full(2 * n + 1, 0.5 / k)
        self.Wm[0] = self.kappa / k

        self.Wc = copy(self.Wm)

    # ------------------------------------------------------------------
    def sigma_points(self, x, P):
        """Return ``(2n+1, n)`` sigma-point matrix.

        Parameters
        ----------
        x:
            State mean, shape ``(n,)``.
        P:
            State covariance, shape ``(n, n)``.
        """
        n = self.n
        k = n + self.kappa

        x = reshape(asarray(x, dtype=float64), (-1,))
        P = asarray(P, dtype=float64)

        U = linalg.cholesky(k * P)  # lower-triangular

        sigmas = empty((2 * n + 1, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1] = x + U[:, i]
            sigmas[n + i + 1] = x - U[:, i]
        return sigmas


# ---------------------------------------------------------------------------
# Unscented Kalman Filter
# ---------------------------------------------------------------------------


class UnscentedKalmanFilter:
    """Minimal Unscented Kalman Filter compatible with the ``filterpy`` API.

    Parameters
    ----------
    model:
        A :class:`_UKFModel` namedtuple carrying ``dim_x``, ``dim_z``,
        ``dt``, ``hx``, ``fx``, and ``points``.
    """

    def __init__(self, model: _UKFModel):
        self._model = model

        self.x = zeros(model.dim_x)
        self.P = eye(model.dim_x)
        self.Q = eye(model.dim_x)
        self.R = eye(model.dim_z)
        self._sigmas_f = None  # populated by predict(), cleared after update()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, fx=None, dt=None, **fx_args):
        """UKF predict step.

        Parameters
        ----------
        fx:
            Override the default state-transition function.
        dt:
            Override the default time step.
        """
        if fx is None:
            fx = self._model.fx
        if dt is None:
            dt = self._model.dt

        points = self._model.points
        sigmas = points.sigma_points(self.x, self.P)
        n_sigmas = sigmas.shape[0]

        sigmas_f = empty_like(sigmas)
        for i in range(n_sigmas):
            sigmas_f[i] = reshape(
                asarray(fx(sigmas[i], dt, **fx_args), dtype=float64), (-1,)
            )

        Wm = points.Wm
        Wc = points.Wc

        x_pred = einsum("i,ij->j", Wm, sigmas_f)

        P_pred = zeros((self._model.dim_x, self._model.dim_x))
        for i in range(n_sigmas):
            d = expand_dims(sigmas_f[i] - x_pred, -1)
            P_pred = P_pred + Wc[i] * (d @ transpose(d))
        P_pred = P_pred + asarray(self.Q, dtype=float64)
        P_pred = 0.5 * (P_pred + transpose(P_pred))

        self.x = x_pred
        self.P = P_pred
        self._sigmas_f = sigmas_f  # store for update()

    def _innovation_matrices(  # pylint: disable=too-many-positional-arguments
        self, sigmas_f, sigmas_h, z_pred, R, Wc
    ):
        """Compute innovation covariance *Pz* and cross-covariance *Pxz*."""
        Pz = zeros((self._model.dim_z, self._model.dim_z))
        Pxz = zeros((self._model.dim_x, self._model.dim_z))
        for i in range(len(Wc)):  # pylint: disable=consider-using-enumerate
            dz = expand_dims(sigmas_h[i] - z_pred, -1)
            dx = expand_dims(sigmas_f[i] - self.x, -1)
            Pz = Pz + Wc[i] * (dz @ transpose(dz))
            Pxz = Pxz + Wc[i] * (dx @ transpose(dz))
        return Pz + R, Pxz

    def update(self, z, R=None, hx=None, **hx_args):
        """UKF update step.

        Parameters
        ----------
        z:
            Measurement vector, shape ``(dim_z,)``.
        R:
            Override measurement noise covariance.
        hx:
            Override the default measurement function.
        """
        if hx is None:
            hx = self._model.hx
        if R is None:
            R = self.R
        R = asarray(R, dtype=float64)
        z = reshape(asarray(z, dtype=float64), (-1,))

        if self._sigmas_f is None:
            self._sigmas_f = self._model.points.sigma_points(self.x, self.P)

        sigmas_f = self._sigmas_f
        Wm = self._model.points.Wm
        Wc = self._model.points.Wc

        sigmas_h = empty((sigmas_f.shape[0], self._model.dim_z))
        for i in range(sigmas_f.shape[0]):
            sigmas_h[i] = reshape(
                asarray(hx(sigmas_f[i], **hx_args), dtype=float64), (-1,)
            )

        z_pred = einsum("i,ij->j", Wm, sigmas_h)
        Pz, Pxz = self._innovation_matrices(sigmas_f, sigmas_h, z_pred, R, Wc)

        # Kalman gain  (solve Pz K^T = Pxz^T  =>  K = (Pz^{-1} Pxz^T)^T)
        K = transpose(linalg.solve(Pz, transpose(Pxz)))

        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ Pz @ transpose(K)
        self.P = 0.5 * (self.P + transpose(self.P))

        self._sigmas_f = None  # clear cached sigma points

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
