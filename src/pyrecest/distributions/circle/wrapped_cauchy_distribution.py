# pylint: disable=no-name-in-module,no-member
from numbers import Integral

import numpy as np
from pyrecest.backend import (
    all,
    arctan2,
    array,
    atleast_1d,
    cos,
    exp,
    isfinite,
    mod,
    pi,
    sin,
    tanh,
)

from .abstract_circular_distribution import AbstractCircularDistribution

_INVALID_REAL_SCALAR_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
    complex,
    np.complexfloating,
    np.datetime64,
    np.timedelta64,
)
_INVALID_REAL_DTYPE_KINDS = {"b", "S", "U", "c", "M", "m"}


def _contains_invalid_real_value(value) -> bool:
    """Return whether a value is boolean, textual, complex, or temporal."""
    if isinstance(value, _INVALID_REAL_SCALAR_TYPES):
        return True

    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        try:
            return np.dtype(dtype).kind in _INVALID_REAL_DTYPE_KINDS
        except (TypeError, ValueError):
            dtype_name = str(dtype).lower()
            return any(
                token in dtype_name
                for token in (
                    "bool",
                    "complex",
                    "str",
                    "bytes",
                    "datetime",
                    "timedelta",
                )
            )

    try:
        values = np.asarray(value, dtype=object).reshape(-1)
    except (OverflowError, TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _INVALID_REAL_SCALAR_TYPES) for item in values)


def _validate_finite_scalar(value, name):
    if _contains_invalid_real_value(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    try:
        value = array(value)
    except Exception as exc:  # pragma: no cover - backend-specific conversion type
        raise ValueError(f"{name} must be a finite real scalar.") from exc
    if value.shape not in ((), (1,)):
        raise ValueError(f"{name} must be a scalar.")
    if value.shape == (1,):
        value = value.reshape(())
    try:
        finite = bool(all(isfinite(value)))
    except (OverflowError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a finite real scalar.") from exc
    if not finite:
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_positive_scalar(value, name):
    value = _validate_finite_scalar(value, name)
    if not bool(all(value > 0.0)):
        raise ValueError(f"{name} must be positive.")
    return value


def _as_1d_input(xs):
    message = "xs must contain only finite real values."
    if _contains_invalid_real_value(xs):
        raise ValueError(message)
    try:
        xs = atleast_1d(array(xs))
    except Exception as exc:  # pragma: no cover - backend-specific conversion type
        raise ValueError(message) from exc
    if xs.ndim != 1:
        raise ValueError("xs must be a one-dimensional array.")
    try:
        finite = bool(all(isfinite(xs)))
    except (OverflowError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(message) from exc
    if not finite:
        raise ValueError(message)
    return xs


class WrappedCauchyDistribution(AbstractCircularDistribution):
    """Wrapped Cauchy distribution on the circle.

    References
    ----------
    Jammalamadaka, S. R., & SenGupta, A. (2001). Topics in Circular
    Statistics. World Scientific.
    """

    def __init__(self, mu, gamma):
        AbstractCircularDistribution.__init__(self)
        self.mu = mod(_validate_finite_scalar(mu, "mu"), 2 * pi)
        self.gamma = _validate_positive_scalar(gamma, "gamma")

    def pdf(self, xs):
        xs = _as_1d_input(xs)
        xs_centered = mod(xs - self.mu, 2 * pi)

        # The usual rho = exp(-gamma) expression subtracts rho from one.
        # For tiny positive gamma, rho rounds to one and that form produces
        # 0 / 0 at the mode. The equivalent half-angle representation avoids
        # this cancellation and is also stable in the large-gamma limit.
        half_angle = xs_centered / 2.0
        half_gamma_tanh = tanh(self.gamma / 2.0)
        denominator = sin(half_angle) ** 2 + half_gamma_tanh**2 * cos(half_angle) ** 2
        return half_gamma_tanh / (2.0 * pi * denominator)

    def cdf(self, xs, starting_point=0.0):
        """
        Evaluate the circular CDF from ``starting_point`` to ``xs``.

        The antiderivative of the wrapped Cauchy density contains
        ``atan(coth(gamma / 2) * tan((x - mu) / 2))``. Evaluating that
        expression directly loses the quadrant information at ``x - mu = pi``.
        Use ``atan2`` on the half-angle representation instead, then subtract
        the value at the requested starting point.
        """

        starting_point = _validate_finite_scalar(starting_point, "starting_point")
        xs = _as_1d_input(xs)
        half_gamma_tanh = tanh(self.gamma / 2.0)

        def primitive(angles):
            angles = array(angles)
            angles_centered = mod(angles - self.mu, 2.0 * pi)
            half_angles = angles_centered / 2.0
            # Multiplying both atan2 arguments by tanh(gamma / 2) preserves
            # the angle while avoiding reciprocal overflow for tiny gamma.
            return (
                arctan2(
                    sin(half_angles),
                    half_gamma_tanh * cos(half_angles),
                )
                / pi
            )

        return mod(primitive(xs) - primitive(starting_point), 1.0)

    def trigonometric_moment(self, n):
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise ValueError("n must be an integer.")
        n = int(n)
        return exp(1j * n * self.mu - abs(n) * self.gamma)
