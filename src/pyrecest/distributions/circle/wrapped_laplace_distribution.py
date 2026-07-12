# pylint: disable=no-name-in-module,no-member
from numbers import Integral

from pyrecest.backend import all, asarray, exp, isfinite, mod, ndim, pi

from .abstract_circular_distribution import AbstractCircularDistribution


def _validate_positive_scalar(value, name):
    value = asarray(value)
    if value.shape not in ((), (1,)):
        raise ValueError(f"{name} must be a positive scalar.")
    if not bool(all(isfinite(value))):
        raise ValueError(f"{name} must be finite.")
    if not bool(all(value > 0.0)):
        raise ValueError(f"{name} must be positive.")
    return value


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
        lambda_ = _validate_positive_scalar(lambda_, "lambda_")
        kappa_ = _validate_positive_scalar(kappa_, "kappa_")
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
        p = (
            self.lambda_
            * self.kappa
            / (1 + self.kappa**2)
            * (
                exp(-positive_rate * xs)
                / (1 - exp(-2.0 * pi * positive_rate))
                + exp(-negative_rate * (2.0 * pi - xs))
                / (1 - exp(-2.0 * pi * negative_rate))
            )
        )
        return p
