# pylint: disable=no-name-in-module,no-member
from numbers import Integral

import numpy as np
from pyrecest.backend import (
    all,
    asarray,
)
from pyrecest.backend import copy as backend_copy
from pyrecest.backend import (
    exp,
    isfinite,
    mod,
    ndim,
    pi,
    to_numpy,
)

from .abstract_circular_distribution import AbstractCircularDistribution

_SMALL_RATE_SERIES_THRESHOLD = 1e-4


def _validate_positive_scalar(value, name):
    message = f"{name} must be a positive real scalar."
    try:
        value = asarray(value)
        numpy_value = np.asarray(to_numpy(value))
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(message) from exc
    if numpy_value.shape not in ((), (1,)) or numpy_value.dtype.kind not in "iuf":
        raise ValueError(message)
    if numpy_value.shape == (1,):
        value = value.reshape(())
    if not bool(all(isfinite(value))):
        raise ValueError(f"{name} must be finite.")
    if not bool(all(value > 0.0)):
        raise ValueError(f"{name} must be positive.")
    return value


def _wrapped_exponential_density(rate, distance):
    """Evaluate a wrapped exponential component without small-rate cancellation."""
    log_beta = 2.0 * pi * rate
    if bool(all(log_beta < _SMALL_RATE_SERIES_THRESHOLD)):
        # rate / (1 - exp(-2*pi*rate)) expanded around rate == 0.
        normalization = (
            1.0 / (2.0 * pi) + rate / 2.0 + pi * rate**2 / 6.0 - pi**3 * rate**4 / 90.0
        )
    else:
        normalization = rate / (1.0 - exp(-log_beta))
    return normalization * exp(-rate * distance)


class WrappedLaplaceDistribution(AbstractCircularDistribution):
    """Wrapped Laplace distribution on the circle.

    References
    ----------
    Jammalamadaka, S. R., & Kozubowski, T. J. (2004). New families of
    wrapped distributions for modeling skew circular data. Communications in
    Statistics - Theory and Methods, 33(9), 2059-2074.
    """

    def __init__(self, lambda_, kappa_):
        AbstractCircularDistribution.__init__(self)
        lambda_ = backend_copy(_validate_positive_scalar(lambda_, "lambda_"))
        kappa_ = backend_copy(_validate_positive_scalar(kappa_, "kappa_"))
        self.lambda_ = lambda_
        self.kappa = kappa_

    def trigonometric_moment(self, n):
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise ValueError("n must be an integer.")
        n = int(n)
        return (
            1
            / (1 - 1j * n / self.lambda_ / self.kappa)
            / (1 + 1j * n / (self.lambda_ / self.kappa))
        )

    def pdf(self, xs):
        xs = asarray(xs)
        if ndim(xs) > 1:
            raise ValueError("xs must be a scalar or one-dimensional array.")
        xs = mod(xs, 2.0 * pi)
        positive_rate = self.lambda_ * self.kappa
        negative_rate = self.lambda_ / self.kappa
        mixture_normalization = 1.0 + self.kappa**2
        p = (
            _wrapped_exponential_density(positive_rate, xs)
            + self.kappa**2 * _wrapped_exponential_density(negative_rate, 2.0 * pi - xs)
        ) / mixture_normalization
        return p
