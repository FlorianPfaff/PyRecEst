# pylint: disable=no-name-in-module,no-member
from numbers import Integral
from typing import Union

import numpy as np

# pylint: disable=redefined-builtin
from pyrecest.backend import (
    all,
    asarray,
)
from pyrecest.backend import copy as backend_copy
from pyrecest.backend import (
    exp,
    int32,
    int64,
    isfinite,
    log,
    log1p,
    mod,
    ndim,
    pi,
    random,
)

from .abstract_circular_distribution import AbstractCircularDistribution

_SMALL_RATE_SERIES_THRESHOLD = 1e-4
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


def _validate_positive_scalar(value, name):
    if _contains_invalid_real_value(value):
        raise ValueError(f"{name} must be a positive real scalar.")
    try:
        value = asarray(value)
    except Exception as exc:  # pragma: no cover - backend-specific conversion type
        raise ValueError(f"{name} must be a positive real scalar.") from exc
    if value.shape not in ((), (1,)):
        raise ValueError(f"{name} must be a positive scalar.")
    if value.shape == (1,):
        value = value.reshape(())
    try:
        finite = bool(all(isfinite(value)))
        positive = bool(all(value > 0.0))
    except (OverflowError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a positive real scalar.") from exc
    if not finite:
        raise ValueError(f"{name} must be finite.")
    if not positive:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_pdf_points(value):
    message = "xs must contain only finite real values."
    if _contains_invalid_real_value(value):
        raise ValueError(message)
    try:
        value = asarray(value)
    except Exception as exc:  # pragma: no cover - backend-specific conversion type
        raise ValueError(message) from exc
    if ndim(value) > 1:
        raise ValueError("xs must be a scalar or one-dimensional array.")
    try:
        finite = bool(all(isfinite(value)))
    except (OverflowError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(message) from exc
    if not finite:
        raise ValueError(message)
    return value


def _density_scale_from_log_beta(log_beta):
    """Return ``lambda / (1 - exp(-2*pi*lambda))`` without overflow."""
    if bool(all(log_beta < _SMALL_RATE_SERIES_THRESHOLD)):
        # x / (1 - exp(-x)) = 1 + x/2 + x**2/12 - x**4/720 + O(x**6).
        return (1.0 + log_beta / 2.0 + log_beta**2 / 12.0 - log_beta**4 / 720.0) / (
            2.0 * pi
        )
    return (log_beta / (2.0 * pi)) / (1.0 - exp(-log_beta))


class WrappedExponentialDistribution(AbstractCircularDistribution):
    """Wrapped exponential distribution on the circle.

    See Sreenivasa Rao Jammalamadaka and Tomasz J. Kozubowski, "New
    Families of Wrapped Distributions for Modeling Skew Circular Data",
    Communications in Statistics - Theory and Methods, Vol. 33, No. 9,
    pp. 2059-2074, 2004.
    """

    def __init__(self, lambda_):
        AbstractCircularDistribution.__init__(self)
        lambda_ = backend_copy(_validate_positive_scalar(lambda_, "lambda_"))
        self.lambda_ = lambda_
        self._log_beta = 2.0 * pi * lambda_
        self._density_scale = _density_scale_from_log_beta(self._log_beta)

    def pdf(self, xs):
        xs = _validate_pdf_points(xs)
        xs = mod(xs, 2.0 * pi)
        return self._density_scale * exp(-self.lambda_ * xs)

    def trigonometric_moment(self, n):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral):
            raise ValueError("n must be an integer.")
        n = int(n)
        return 1.0 / (1.0 - 1j * n / self.lambda_)

    def sample(self, n: Union[int, int32, int64]):
        if isinstance(n, bool) or not isinstance(n, Integral) or int(n) <= 0:
            raise ValueError("n must be a positive integer.")
        n = int(n)
        # Use inverse CDF method: X = -ln(U)/lambda ~ Exp(lambda), then wrap
        u = random.uniform(size=(n,))
        u = u + (u == 0.0) * 1.0e-12
        return mod(-log(u) / self.lambda_, 2.0 * pi)

    def entropy(self):
        log_beta = self._log_beta
        if bool(all(log_beta < _SMALL_RATE_SERIES_THRESHOLD)):
            # As lambda approaches zero, the distribution approaches the uniform
            # distribution on [0, 2*pi).  The direct expression evaluates
            # log1p(-exp(-log_beta)) and divides by 1 - exp(-log_beta), which
            # suffers catastrophic cancellation for tiny log_beta.
            return log(2.0 * pi) - log_beta**2 / 24.0 + log_beta**4 / 960.0

        # Use exp(-2*pi*lambda) to avoid overflowing exp(2*pi*lambda) for
        # concentrated wrapped exponentials.
        exp_neg_log_beta = exp(-log_beta)
        return (
            1.0
            - log(self.lambda_)
            + log1p(-exp_neg_log_beta)
            - log_beta * exp_neg_log_beta / (1.0 - exp_neg_log_beta)
        )
