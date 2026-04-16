"""
Unscented Kalman Filter on (parallelizable) Manifolds (UKF-M).

Based on the algorithm described in:
    Martin Brossard, Axel Barrau, Silvère Bonnabel,
    "A Code for Unscented Kalman Filtering on Manifolds (UKF-M)", 2019.

Reference Python implementation:
    https://github.com/CAOR-MINES-ParisTech/ukfm
"""

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from copy import copy
from typing import Any, Callable

import pyrecest.backend
from pyrecest.backend import (
    asarray,
    broadcast_to,
    eye,
    linalg,
    outer,
    sqrt,
    sum,
    vstack,
    zeros,
)

from .abstract_filter import AbstractFilter


class _Weights:
    """Sigma-point weights for UKF-M."""

    def __init__(self, d: int, alpha: float):
        m = (alpha**2 - 1) * d
        self.sqrt_d_lambda = sqrt(d + m)
        self.wj = 1.0 / (2.0 * (d + m))
        self.wm = m / (m + d)
        self.w0 = m / (m + d) + 3.0 - alpha**2


class UKFOnManifolds(AbstractFilter):  # pylint: disable=too-many-instance-attributes
    """Unscented Kalman Filter on (parallelizable) Manifolds.

    Implements the UKF-M algorithm that works for states living on smooth
    manifolds.  The uncertainty is represented as a covariance matrix in the
    tangent space at the current state estimate.

    The state can be any Python object (e.g. a numpy array, a rotation matrix,
    a tuple representing a Lie group element).  All manifold-specific
    operations are provided by the user via the ``phi``, ``phi_inv``
    callables.

    Parameters
    ----------
    f:
        Propagation (process) function with signature
        ``f(state, omega, noise, dt) -> new_state``.
        ``noise`` is a 1-D numpy array of length ``q`` (noise dimension).
    h:
        Observation function with signature ``h(state) -> y`` where ``y`` is
        a 1-D numpy array of length ``l``.
    phi:
        Retraction (exponential-like map) with signature
        ``phi(state, xi) -> new_state``.  ``xi`` is a 1-D numpy array in
        the tangent space (length ``d``).
    phi_inv:
        Inverse retraction with signature
        ``phi_inv(state_ref, state) -> xi``.  Returns a 1-D numpy array.
    Q:
        Process noise covariance matrix, shape ``(q, q)``.
    R:
        Measurement noise covariance matrix, shape ``(l, l)``.
    alpha:
        Sigma-point spread parameters.  Either a scalar (same value used for
        all three weight sets) or a length-3 array-like
        ``[alpha_d, alpha_q, alpha_u]`` where:

        * ``alpha_d`` — propagation w.r.t. state uncertainty,
        * ``alpha_q`` — propagation w.r.t. noise,
        * ``alpha_u`` — update.

        Typical value: ``1e-3``.
    state0:
        Initial state estimate (manifold element).
    P0:
        Initial covariance matrix, shape ``(d, d)``, where ``d`` is the
        dimension of the tangent space / uncertainty.

    Examples
    --------
    A simple Euclidean example (identity manifold, phi = state + xi):

    >>> from pyrecest.backend import array, eye, zeros
    >>> from pyrecest.filters import UKFOnManifolds
    >>> f = lambda s, omega, w, dt: s + omega * dt + w
    >>> h = lambda s: s
    >>> phi = lambda s, xi: s + xi
    >>> phi_inv = lambda s_ref, s: s - s_ref
    >>> Q = eye(2) * 0.1
    >>> R = eye(2) * 0.5
    >>> s0 = zeros(2)
    >>> P0 = eye(2)
    >>> ukf = UKFOnManifolds(f, h, phi, phi_inv, Q, R, 1e-3, s0, P0)
    >>> ukf.predict(omega=zeros(2), dt=1.0)
    >>> ukf.update(y=array([1.0, 0.5]))
    """

    TOL = 1e-9  # tolerance for ensuring positive-definiteness

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        f: Callable,
        h: Callable,
        phi: Callable,
        phi_inv: Callable,
        Q,
        R,
        alpha,
        state0: Any,
        P0,
    ):
        Q = asarray(Q)
        R = asarray(R)
        P0 = asarray(P0)

        self.f = f
        self.h = h
        self.phi = phi
        self.phi_inv = phi_inv
        self.Q = Q
        self.R = R

        # Cholesky factor of Q (rows are columns of L^T so cholQ[j] gives the
        # j-th column of L^T, matching the reference implementation)
        self.cholQ = linalg.cholesky(Q).T

        # Dimensions
        self.d = P0.shape[0]  # tangent-space dimension
        self.q = Q.shape[0]  # noise dimension
        self.meas_dim = R.shape[0]  # measurement dimension

        # Sigma-point weights — three sets with potentially different alphas
        # Handle scalar or length-3 array-like for alpha
        try:
            alpha_list = list(alpha)
            if len(alpha_list) != 3:
                raise ValueError("alpha must be a scalar or a length-3 array-like")
        except TypeError:
            alpha_list = [float(alpha)] * 3
        alpha_arr = broadcast_to(asarray(alpha_list), (3,))
        self._w_d = _Weights(self.d, float(alpha_arr[0]))  # propagation / state
        self._w_q = _Weights(self.q, float(alpha_arr[1]))  # propagation / noise
        self._w_u = _Weights(self.d, float(alpha_arr[2]))  # update

        self._state = state0
        self._P = copy(P0)

        # AbstractFilter stores the filter state; we use a tuple (state, P)
        AbstractFilter.__init__(self, (self._state, self._P))

    # ------------------------------------------------------------------
    # filter_state property – expose as (state, P) tuple
    # ------------------------------------------------------------------

    @property
    def filter_state(self):
        """Return the current filter state as a ``(state, P)`` tuple."""
        return (self._state, self._P)

    @filter_state.setter
    def filter_state(self, new_state):
        if isinstance(new_state, tuple) and len(new_state) == 2:
            self._state, P = new_state
            self._P = asarray(P)
        else:
            raise ValueError("filter_state must be a (state, covariance) tuple")
        # Keep AbstractFilter's internal reference consistent
        self._filter_state = (self._state, self._P)

    # ------------------------------------------------------------------
    # Prediction / propagation
    # ------------------------------------------------------------------

    def predict(self, omega=None, dt: float = 1.0):  # pylint: disable=too-many-locals
        """Propagate the filter state.

        Parameters
        ----------
        omega:
            Control / input passed to ``f``.  Set to ``None`` if ``f`` does
            not use it (the filter passes it through).
        dt:
            Integration step (seconds).
        """
        if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
            raise NotImplementedError("predict is not supported on the JAX backend.")
        P = self._P + self.TOL * eye(self.d)

        w_zero = zeros(self.q)

        # 1. Propagate the mean
        new_state = self.f(self._state, omega, w_zero, dt)

        # 2. Covariance contribution from state uncertainty
        w_d = self._w_d
        chol_P = linalg.cholesky(P).T  # rows = columns of upper Chol.
        xis = w_d.sqrt_d_lambda * chol_P  # shape (d, d)

        new_xis = zeros((2 * self.d, self.d))
        for j in range(self.d):
            s_plus = self.phi(self._state, xis[j])
            s_minus = self.phi(self._state, -xis[j])
            new_xis[j] = self.phi_inv(new_state, self.f(s_plus, omega, w_zero, dt))
            new_xis[self.d + j] = self.phi_inv(
                new_state, self.f(s_minus, omega, w_zero, dt)
            )

        xi_mean = w_d.wj * sum(new_xis, axis=0)
        new_xis_centered = new_xis - xi_mean
        new_P = w_d.wj * new_xis_centered.T @ new_xis_centered + w_d.w0 * outer(
            xi_mean, xi_mean
        )

        # 3. Covariance contribution from noise
        w_q = self._w_q
        new_xis_q = zeros((2 * self.q, self.d))
        for j in range(self.q):
            w_plus = w_q.sqrt_d_lambda * self.cholQ[j]
            w_minus = -w_q.sqrt_d_lambda * self.cholQ[j]
            new_xis_q[j] = self.phi_inv(
                new_state, self.f(self._state, omega, w_plus, dt)
            )
            new_xis_q[self.q + j] = self.phi_inv(
                new_state, self.f(self._state, omega, w_minus, dt)
            )

        xi_mean_q = w_q.wj * sum(new_xis_q, axis=0)
        new_xis_q_centered = new_xis_q - xi_mean_q
        Q_contrib = w_q.wj * new_xis_q_centered.T @ new_xis_q_centered + w_q.w0 * outer(
            xi_mean_q, xi_mean_q
        )

        self._P = new_P + Q_contrib
        self._state = new_state
        self._filter_state = (self._state, self._P)

    # Alias to match naming convention of other filters
    predict_nonlinear = predict

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, y):  # pylint: disable=too-many-locals
        """Update the filter with a new measurement.

        Parameters
        ----------
        y:
            1-D measurement vector, shape ``(l,)``.
        """
        if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
            raise NotImplementedError("update is not supported on the JAX backend.")
        y = asarray(y).ravel()
        P = self._P + self.TOL * eye(self.d)

        w_u = self._w_u
        chol_P = linalg.cholesky(P).T
        xis = w_u.sqrt_d_lambda * chol_P  # shape (d, d)

        # Compute predicted measurements at sigma points
        ys = zeros((2 * self.d, self.meas_dim))
        hat_y = asarray(self.h(self._state)).ravel()
        for j in range(self.d):
            s_plus = self.phi(self._state, xis[j])
            s_minus = self.phi(self._state, -xis[j])
            ys[j] = asarray(self.h(s_plus)).ravel()
            ys[self.d + j] = asarray(self.h(s_minus)).ravel()

        # Predicted measurement mean
        y_bar = w_u.wm * hat_y + w_u.wj * sum(ys, axis=0)

        # Centered residuals
        ys_centered = ys - y_bar
        hat_y_centered = hat_y - y_bar

        # Innovation covariance and cross-covariance
        P_yy = (
            w_u.w0 * outer(hat_y_centered, hat_y_centered)
            + w_u.wj * ys_centered.T @ ys_centered
            + self.R
        )
        # Cross-covariance: shape (d, l)
        # xis rows: [xi_1, ..., xi_d, -xi_1, ..., -xi_d]  (sign already in ys)
        xi_stack = vstack([xis, -xis])  # (2d, d)
        P_xiy = w_u.wj * xi_stack.T @ ys_centered  # (d, l)

        # Kalman gain
        K = linalg.solve(P_yy, P_xiy.T).T  # (d, l)

        # State update
        xi_plus = K @ (y - y_bar)
        self._state = self.phi(self._state, xi_plus)

        # Covariance update
        self._P = P - K @ P_yy @ K.T
        self._P = (self._P + self._P.T) / 2.0
        self._filter_state = (self._state, self._P)

    # ------------------------------------------------------------------
    # Point estimate
    # ------------------------------------------------------------------

    def get_point_estimate(self):
        """Return the current state estimate (manifold element)."""
        return self._state
