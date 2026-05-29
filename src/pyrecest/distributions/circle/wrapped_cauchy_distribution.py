# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    all,
    arctan2,
    array,
    atleast_1d,
    cos,
    cosh,
    exp,
    isfinite,
    mod,
    pi,
    sin,
    sinh,
    tanh,
)

from .abstract_circular_distribution import AbstractCircularDistribution


def _validate_positive_scalar(value, name):
    value = array(value)
    if value.shape not in ((), (1,)):
        raise ValueError(f"{name} must be a positive scalar.")
    if not bool(all(isfinite(value))):
        raise ValueError(f"{name} must be finite.")
    if not bool(all(value > 0.0)):
        raise ValueError(f"{name} must be positive.")
    return value


def _as_1d_input(xs):
    xs = atleast_1d(array(xs))
    if xs.ndim != 1:
        raise ValueError("xs must be a one-dimensional array.")
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
        self.mu = mod(mu, 2 * pi)
        self.gamma = _validate_positive_scalar(gamma, "gamma")

    def pdf(self, xs):
        xs = _as_1d_input(xs)
        xs_centered = mod(xs - self.mu, 2 * pi)
        return 1 / (2 * pi) * sinh(self.gamma) / (cosh(self.gamma) - cos(xs_centered))

    def cdf(self, xs, starting_point=0.0):
        """
        Evaluate the circular CDF from ``starting_point`` to ``xs``.

        The antiderivative of the wrapped Cauchy density contains
        ``atan(coth(gamma / 2) * tan((x - mu) / 2))``. Evaluating that
        expression directly loses the quadrant information at ``x - mu = pi``.
        Use ``atan2`` on the half-angle representation instead, then subtract
        the value at the requested starting point.
        """

        def coth(x):
            return 1 / tanh(x)

        xs = _as_1d_input(xs)

        def primitive(angles):
            angles = array(angles)
            angles_centered = mod(angles - self.mu, 2.0 * pi)
            half_angles = angles_centered / 2.0
            return (
                arctan2(coth(self.gamma / 2.0) * sin(half_angles), cos(half_angles))
                / pi
            )

        return mod(primitive(xs) - primitive(starting_point), 1.0)

    def trigonometric_moment(self, n):
        m = exp(1j * n * self.mu - abs(n) * self.gamma)
        return m
