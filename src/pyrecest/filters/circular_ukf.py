"""
A modified unscented Kalman filter for circular distributions,
interprets circle as 1D interval [0, 2*pi).

References:
    Gerhard Kurz, Igor Gilitschenski, Uwe D. Hanebeck,
    Recursive Bayesian Filtering in Circular State Spaces
    arXiv preprint: Systems and Control (cs.SY), January 2015.
"""

import math

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    array,
    asarray,
    atleast_1d,
    empty,
    float64,
    isfinite,
    linalg,
    mod,
    pi,
    reshape,
    sign,
    transpose,
    zeros,
)
from pyrecest.distributions import GaussianDistribution
from pyrecest.sampling.sigma_points import MerweScaledSigmaPoints

from .abstract_filter import AbstractFilter
from .manifold_mixins import CircularFilterMixin

_TWO_PI = 2.0 * float(pi)


def _to_python_bool(value):
    """Convert scalar backend booleans to Python bools for validation."""
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _as_circular_gaussian(distribution, role):
    """Return a validated 1-D Gaussian for circular UKF state/noise."""
    if not isinstance(distribution, GaussianDistribution):
        try:
            distribution = GaussianDistribution.from_distribution(distribution)
        except Exception as exc:  # pragma: no cover - backend-specific failures
            raise ValueError(
                f"{role} must be convertible to GaussianDistribution."
            ) from exc

    mu = asarray(distribution.mu)
    covariance = asarray(distribution.C)
    if distribution.dim != 1 or mu.shape != (1,):
        raise ValueError(f"{role} mean must be one-dimensional.")
    if covariance.shape != (1, 1):
        raise ValueError(f"{role} covariance must have shape (1, 1).")
    if not _to_python_bool(backend_all(isfinite(mu))):
        raise ValueError(f"{role} mean must be finite.")
    if not _to_python_bool(backend_all(isfinite(covariance))):
        raise ValueError(f"{role} covariance must be finite.")
    if float(covariance[0, 0]) <= 0.0:
        raise ValueError(f"{role} covariance must be positive.")
    return distribution


def _validate_backend_supported(operation):
    if pyrecest.backend.__backend_name__ in ("pytorch", "jax"):
        raise NotImplementedError(f"{operation} is not supported on this backend.")


def _validate_circular_scalar_measurement(z):
    measurement = array(z, dtype=float)
    if measurement.shape not in ((), (1,)):
        raise ValueError("measurement z must be scalar.")
    measurement = measurement[0] if measurement.shape == (1,) else measurement
    if not _to_python_bool(isfinite(measurement)):
        raise ValueError("measurement z must be finite.")
    return float(measurement)


def _measurement_vector(value):
    """Convert a scalar or vector-valued measurement to a flat 1-D array."""
    measurement = atleast_1d(array(value, dtype=float)).flatten()
    if measurement.shape[0] == 0:
        raise ValueError("measurement vector must not be empty.")
    if not _to_python_bool(backend_all(isfinite(measurement))):
        raise ValueError("measurement vector must be finite.")
    return measurement


def _validate_measurement_noise_vector(noise_mean, dim_z):
    if len(noise_mean) != dim_z:
        raise ValueError(
            "measurement noise mean dimension mismatch: z has dimension "
            f"{dim_z}, but gauss_meas.mu has dimension {len(noise_mean)}"
        )


def _wrap_angle_scalar(value):
    """Map a scalar angle to [0, 2*pi)."""
    return float(value) % _TWO_PI


def _angular_difference_scalar(value, reference):
    """Return the signed angle from reference to value in [-pi, pi)."""
    return (float(value) - float(reference) + float(pi)) % _TWO_PI - float(pi)


def _periodic_difference(value, reference):
    """Return element-wise signed periodic differences in [-pi, pi)."""
    return mod(value - reference + pi, 2.0 * pi) - pi


def _wrap_periodic_measurement_to_reference(value, reference):
    """Put periodic measurement components on the branch nearest reference."""
    return reference + _periodic_difference(value, reference)


def _weighted_circular_mean(angles, weights, reference=None):
    """Local weighted circular mean for sigma points.

    Merwe sigma-point weights can be negative.  A local lift around the central
    sigma point is therefore more stable than a phasor mean when all sigma
    points lie inside one local branch, which is the regime assumed by this UKF.
    """
    if reference is None:
        reference = float(angles[0])
    mean_lifted = float(reference) + math.fsum(
        float(w) * _angular_difference_scalar(angle, reference)
        for w, angle in zip(weights, angles)
    )
    return _wrap_angle_scalar(mean_lifted)


def _sigma_points_1d(mu, covariance, alpha, beta, kappa):
    """Return 1-D Merwe sigma points and their weights."""
    points = MerweScaledSigmaPoints(n=1, alpha=alpha, beta=beta, kappa=kappa)
    sigmas = points.sigma_points(array([mu]), array([[covariance]])).flatten()
    return points, sigmas


def _positive_variance(value):
    """Remove tiny negative round-off from scalar covariance updates."""
    value = float(value)
    if value <= 0.0 and abs(value) < 1e-12:
        return 1e-15
    return value


class CircularUKF(AbstractFilter, CircularFilterMixin):
    """
    A modified unscented Kalman filter for circular distributions.

    The state is represented as a 1-D :class:`GaussianDistribution` on the
    circle [0, 2*pi).
    """

    def __init__(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        """Initialise with a standard Gaussian at 0 with unit variance.

        Parameters
        ----------
        alpha:
            UKF sigma-point spread parameter (default 1e-3).
        beta:
            UKF prior distribution parameter (default 2.0, optimal for Gaussian).
        kappa:
            UKF secondary scaling parameter (default 0.0).
        """
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        initial_state = GaussianDistribution(array([0.0]), array([[1.0]]))
        CircularFilterMixin.__init__(self)
        AbstractFilter.__init__(self, initial_state)

    # ------------------------------------------------------------------
    # filter_state property
    # ------------------------------------------------------------------

    @property
    def filter_state(self) -> GaussianDistribution:
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        new_state = _as_circular_gaussian(new_state, "filter_state")
        self._filter_state = new_state

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_identity(self, gauss_sys: GaussianDistribution):
        """
        Predict assuming identity system model:
            x(k+1) = x(k) + w(k)  mod 2*pi
        where *w(k)* is additive noise described by *gauss_sys*.

        Parameters
        ----------
        gauss_sys:
            Distribution of additive system noise (converted to
            :class:`GaussianDistribution` if necessary).
        """
        gauss_sys = _as_circular_gaussian(gauss_sys, "system noise")
        new_mu = mod(self._filter_state.mu + gauss_sys.mu, 2.0 * pi)
        new_C = self._filter_state.C + gauss_sys.C
        self._filter_state = GaussianDistribution(new_mu, new_C)

    def predict_nonlinear(self, f, gauss_sys: GaussianDistribution):
        """
        Predict assuming a nonlinear system model:
            x(k+1) = f(x(k)) + w(k)  mod 2*pi
        where *w(k)* is additive noise described by *gauss_sys*.

        Parameters
        ----------
        f:
            Function from [0, 2*pi) to [0, 2*pi).
        gauss_sys:
            Distribution of additive system noise.
        """
        gauss_sys = _as_circular_gaussian(gauss_sys, "system noise")
        if not callable(f):
            raise ValueError("system function must be callable.")
        _validate_backend_supported("CircularUKF.predict_nonlinear")

        mu0 = float(self._filter_state.mu[0])
        C0 = float(self._filter_state.C[0, 0])
        Q_val = float(gauss_sys.C[0, 0])
        noise_mean = float(gauss_sys.mu[0])

        points, sigmas = _sigma_points_1d(mu0, C0, self._alpha, self._beta, self._kappa)
        propagated = [
            _wrap_angle_scalar(f(_wrap_angle_scalar(sigma)) + noise_mean)
            for sigma in sigmas
        ]

        new_mu_scalar = _weighted_circular_mean(propagated, points.Wm)
        new_C_scalar = math.fsum(
            float(w) * _angular_difference_scalar(angle, new_mu_scalar) ** 2
            for w, angle in zip(points.Wc, propagated)
        )
        new_C_scalar = _positive_variance(new_C_scalar + Q_val)

        self._filter_state = GaussianDistribution(
            array([new_mu_scalar]), array([[new_C_scalar]])
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_identity(self, gauss_meas: GaussianDistribution, z):
        """
        Update assuming identity measurement model:
            z(k) = x(k) + v(k)  mod 2*pi
        where *v(k)* is additive noise described by *gauss_meas*.

        Parameters
        ----------
        gauss_meas:
            Distribution of additive measurement noise.
        z:
            Scalar measurement in [0, 2*pi).
        """
        gauss_meas = _as_circular_gaussian(gauss_meas, "measurement noise")

        z_val = _validate_circular_scalar_measurement(z)
        # Shift measurement by noise mean
        z_val = float(mod(array([z_val - float(gauss_meas.mu[0])]), 2.0 * pi)[0])

        mu = float(self._filter_state.mu[0])
        # Move measurement if necessary (wrap to be nearest to current mean)
        if abs(mu - z_val) > float(pi):
            z_val = z_val + 2.0 * float(pi) * sign(mu - z_val)

        C = float(self._filter_state.C[0, 0])
        R = float(gauss_meas.C[0, 0])

        # Kalman update
        K = C / (C + R)
        new_mu = mu + K * (z_val - mu)
        new_C = (1.0 - K) * C

        new_mu = float(mod(array([new_mu]), 2.0 * pi)[0])
        self._filter_state = GaussianDistribution(array([new_mu]), array([[new_C]]))

    def update_nonlinear(  # pylint: disable=too-many-locals
        self, f, gauss_meas: GaussianDistribution, z, measurement_periodic: bool = False
    ):
        """
        Update assuming a nonlinear measurement model:
            z(k) = f(x(k)) + v(k)           (if *measurement_periodic* is False)
            z(k) = f(x(k)) + v(k)  mod 2*pi (if *measurement_periodic* is True)
        where *v(k)* is additive noise described by *gauss_meas*.

        Parameters
        ----------
        f:
            Measurement function from [0, 2*pi) to R^d.
        gauss_meas:
            Distribution of additive measurement noise.
        z:
            Measurement vector of shape (d,).
        measurement_periodic:
            Whether the measurement is a periodic quantity.
        """
        if not callable(f):
            raise ValueError("measurement function must be callable.")
        if not isinstance(gauss_meas, GaussianDistribution):
            try:
                gauss_meas = GaussianDistribution.from_distribution(gauss_meas)
            except Exception as exc:  # pragma: no cover - backend-specific failures
                raise ValueError(
                    "measurement noise must be convertible to GaussianDistribution."
                ) from exc
        _validate_backend_supported("CircularUKF.update_nonlinear")

        z = _measurement_vector(z)
        dim_z = len(z)
        noise_mean = _measurement_vector(gauss_meas.mu)
        _validate_measurement_noise_vector(noise_mean, dim_z)
        if not _to_python_bool(backend_all(isfinite(gauss_meas.C))):
            raise ValueError("measurement noise covariance must be finite.")

        mu0 = float(self._filter_state.mu[0])
        C0 = float(self._filter_state.C[0, 0])

        if measurement_periodic:
            z_reference = _measurement_vector(f(_wrap_angle_scalar(mu0)))
            if len(z_reference) != dim_z:
                raise ValueError(
                    "measurement dimension mismatch: z has dimension "
                    f"{dim_z}, but f(mu) returns dimension {len(z_reference)}"
                )
            z_reference = z_reference + noise_mean
            z = _wrap_periodic_measurement_to_reference(z, z_reference)
        else:
            z_reference = None

        if dim_z == 1:
            R_mat = array([[float(gauss_meas.C.flatten()[0])]])
        else:
            R_mat = array(gauss_meas.C, dtype=float).reshape(dim_z, dim_z)

        points, sigmas = _sigma_points_1d(mu0, C0, self._alpha, self._beta, self._kappa)
        sigmas_x = [_wrap_angle_scalar(sigma) for sigma in sigmas]
        sigmas_h = empty((len(sigmas_x), dim_z))
        for i, sigma_x in enumerate(sigmas_x):
            hx_val = _measurement_vector(f(sigma_x))
            if len(hx_val) != dim_z:
                raise ValueError(
                    "measurement function must return vectors with consistent "
                    f"dimension; got {len(hx_val)} and expected {dim_z}"
                )
            # R_mat describes only the centered covariance of v; include E[v]
            # in the deterministic measurement model E[z|x] = f(x) + E[v].
            hx_val = hx_val + noise_mean
            if measurement_periodic:
                hx_val = _wrap_periodic_measurement_to_reference(hx_val, z_reference)
            sigmas_h[i] = hx_val

        if measurement_periodic:
            z_pred_values = []
            for dim in range(dim_z):
                z_pred_values.append(
                    _weighted_circular_mean(
                        sigmas_h[:, dim], points.Wm, reference=z_reference[dim]
                    )
                )
            z_pred = array(z_pred_values)
        else:
            z_pred = asarray(points.Wm @ sigmas_h, dtype=float64)

        Pz = zeros((dim_z, dim_z))
        Pxz = zeros((1, dim_z))
        for i, sigma_x in enumerate(sigmas_x):
            if measurement_periodic:
                dz = _periodic_difference(sigmas_h[i], z_pred)
            else:
                dz = sigmas_h[i] - z_pred
            dz = reshape(asarray(dz, dtype=float64), (-1,))
            dz_col = reshape(dz, (-1, 1))
            dx = _angular_difference_scalar(sigma_x, mu0)
            Pz = Pz + float(points.Wc[i]) * (dz_col @ transpose(dz_col))
            Pxz = Pxz + float(points.Wc[i]) * dx * reshape(dz, (1, dim_z))
        Pz = Pz + R_mat

        # Kalman gain  (solve Pz K^T = Pxz^T  =>  K = (Pz^{-1} Pxz^T)^T)
        K = transpose(linalg.solve(Pz, transpose(Pxz)))

        if measurement_periodic:
            innovation = _periodic_difference(z, z_pred)
        else:
            innovation = z - z_pred
        innovation = reshape(asarray(innovation, dtype=float64), (-1,))

        new_mu = _wrap_angle_scalar(mu0 + float((K @ innovation)[0]))
        new_C = array([[C0]]) - K @ Pz @ transpose(K)
        new_C = 0.5 * (new_C + transpose(new_C))
        new_C_scalar = _positive_variance(new_C[0, 0])
        self._filter_state = GaussianDistribution(
            array([new_mu]), array([[new_C_scalar]])
        )

    # ------------------------------------------------------------------
    # Point estimate
    # ------------------------------------------------------------------

    def get_point_estimate(self):
        """Return the mean of the current state estimate."""
        return self._filter_state.mu
