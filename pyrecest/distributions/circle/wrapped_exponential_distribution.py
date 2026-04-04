# pylint: disable=no-name-in-module,no-member
from typing import Union

# pylint: disable=redefined-builtin
from pyrecest.backend import exp, int32, int64, log, mod, ndim, pi, random

from .abstract_circular_distribution import AbstractCircularDistribution


class WrappedExponentialDistribution(AbstractCircularDistribution):
    """Wrapped exponential distribution on the circle.

    See Sreenivasa Rao Jammalamadaka and Tomasz J. Kozubowski, "New
    Families of Wrapped Distributions for Modeling Skew Circular Data",
    Communications in Statistics - Theory and Methods, Vol. 33, No. 9,
    pp. 2059-2074, 2004.
    """

    def __init__(self, lambda_):
        AbstractCircularDistribution.__init__(self)
        assert lambda_.shape in ((1,), ())
        assert lambda_ > 0.0
        self.lambda_ = lambda_
        self._normalization_const = 1.0 / (1.0 - exp(-2.0 * pi * lambda_))

    def pdf(self, xs):
        assert ndim(xs) <= 1
        xs = mod(xs, 2.0 * pi)
        return self.lambda_ * exp(-self.lambda_ * xs) * self._normalization_const

    def trigonometric_moment(self, n):
        return 1.0 / (1.0 - 1j * n / self.lambda_)

    def sample(self, n: Union[int, int32, int64]):
        # Use inverse CDF method: X = -ln(U)/lambda ~ Exp(lambda), then wrap
        u = random.uniform(size=(n,))
        return mod(-log(u) / self.lambda_, 2.0 * pi)

    def entropy(self):
        # log(exp(2*pi*lambda)) = 2*pi*lambda, avoiding redundant exp/log
        log_beta = 2.0 * pi * self.lambda_
        beta = exp(log_beta)
        return 1.0 + log((beta - 1.0) / self.lambda_) - beta / (beta - 1.0) * log_beta
