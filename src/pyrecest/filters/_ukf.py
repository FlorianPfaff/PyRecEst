"""
Unscented Kalman Filter (UKF), replacing the former dependency on the
``bayesian_filters`` package.
"""

from collections import namedtuple
from copy import deepcopy

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    asarray,
    einsum,
    empty,
    empty_like,
    expand_dims,
    eye,
    float64,
    linalg,
    reshape,
    transpose,
    zeros,
)
from pyrecest.sampling.sigma_points import JulierSigmaPoints, MerweScaledSigmaPoints

# ---------------------------------------------------------------------------
# Model configuration container
# ---------------------------------------------------------------------------

_UKFModel = namedtuple("_UKFModel", ["dim_x", "dim_z", "dt", "hx", "fx", "points"])

__all__ = [
    "_UKFModel",
    "JulierSigmaPoints",
    "MerweScaledSigmaPoints",
    "UnscentedKalmanFilter",
]


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
