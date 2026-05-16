# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    arctan2,
    array,
    cos,
    exp,
    imag,
    log,
    mod,
    pi,
    real,
    sin,
    sqrt,
    where,
    zeros_like,
)
from scipy.optimize import fsolve
from scipy.special import iv
from scipy.stats import vonmises

from .abstract_circular_distribution import AbstractCircularDistribution


class VonMisesDistribution(AbstractCircularDistribution):
    """Von Mises distribution on the circle.

    References
    ----------
    Jammalamadaka, S. R., & SenGupta, A. (2001). Topics in Circular
    Statistics. World Scientific.
    """

    def __init__(
        self,
        mu,
        kappa,
        norm_const: float | None = None,
    ):
        assert kappa >= 0.0
        super().__init__()
        self.mu = mu
        self.kappa = kappa
        self._norm_const = norm_const

    def get_params(self):
        return self.mu, self.kappa

    @property
    def norm_const(self):
        if self._norm_const is None:
            self._norm_const = 2.0 * pi * iv(0, self.kappa)
        return self._norm_const

    def pdf(self, xs):
        p = exp(self.kappa * cos(xs - self.mu)) / self.norm_const
        return p

    def sample(self, n):
        """Draw samples from the von Mises distribution."""
        return mod(
            array(
                vonmises.rvs(kappa=float(self.kappa), loc=float(self.mu), size=int(n))
            ),
            2.0 * pi,
        )

    def set_mean(self, mu):
        """
        Set the mean direction of the distribution.

        Parameters:
        mu : scalar
            New mean direction to set.
        """
        self.mu = mu

    @staticmethod
    def besselratio(nu, kappa):
        return iv(nu + 1, kappa) / iv(nu, kappa)

    def cdf(self, xs, starting_point=0):
        """
        Evaluate cumulative distribution function

        Parameters:
        xs : (n)
            points where the cdf should be evaluated
        starting_point : scalar, optional, default: 0
            point where the cdf is zero (starting point can be
            [0, 2pi) on the circle, default is 0)

        Returns:
        val : (n)
            cdf evaluated at columns of xs
        """
        assert xs.ndim <= 1

        r = zeros_like(xs)

        def to_minus_pi_to_pi_range(angle):
            return mod(angle + pi, 2 * pi) - pi

        r = vonmises.cdf(
            to_minus_pi_to_pi_range(xs),
            kappa=self.kappa,
            loc=to_minus_pi_to_pi_range(self.mu),
        ) - vonmises.cdf(
            to_minus_pi_to_pi_range(starting_point),
            kappa=self.kappa,
            loc=to_minus_pi_to_pi_range(self.mu),
        )

        r = where(
            to_minus_pi_to_pi_range(xs) < to_minus_pi_to_pi_range(starting_point),
            1 + r,
            r,
        )
        return r

    @staticmethod
    def besselratio_inverse(v, x):
        def f(t: float) -> float:
            return VonMisesDistribution.besselratio(v, t) - x

        start = 1.0
        (kappa,) = fsolve(f, start)
        return kappa

    def multiply(self, vm2: "VonMisesDistribution") -> "VonMisesDistribution":
        C = self.kappa * cos(self.mu) + vm2.kappa * cos(vm2.mu)
        S = self.kappa * sin(self.mu) + vm2.kappa * sin(vm2.mu)
        mu_ = mod(arctan2(S, C), 2 * pi)
        kappa_ = sqrt(C**2 + S**2)
        return VonMisesDistribution(mu_, kappa_)

    def convolve(self, vm2: "VonMisesDistribution") -> "VonMisesDistribution":
        mu_ = mod(self.mu + vm2.mu, 2.0 * pi)
        t = VonMisesDistribution.besselratio(
            0, self.kappa
        ) * VonMisesDistribution.besselratio(0, vm2.kappa)
        kappa_ = VonMisesDistribution.besselratio_inverse(0, t)
        return VonMisesDistribution(mu_, kappa_)

    def entropy(self):
        result = -self.kappa * VonMisesDistribution.besselratio(0, self.kappa) + log(
            2.0 * pi * iv(0, self.kappa)
        )
        return result

    def trigonometric_moment(self, n: int):
        if n in (0, 1, 2):
            return self.trigonometric_moment_analytic(n)
        return self.trigonometric_moment_numerical(n)

    def trigonometric_moment_analytic(self, n: int):
        if n == 0:
            m = array(1.0 + 0.0j)
        elif self.kappa == 0.0:
            m = array(0.0 + 0.0j)
        elif n == 1:
            m = VonMisesDistribution.besselratio(0, self.kappa) * exp(1j * n * self.mu)
        elif n == 2:
            m = (
                VonMisesDistribution.besselratio(0, self.kappa)
                * VonMisesDistribution.besselratio(1, self.kappa)
                * exp(1j * n * self.mu)
            )
        else:
            raise NotImplementedError("Not implemented")

        return m

    @staticmethod
    def from_moment(m):
        """
        Obtain a VM distribution from a given first trigonometric moment.

        Parameters:
            m (scalar): First trigonometric moment (complex number).

        Returns:
            vm (VMDistribution): VM distribution obtained by moment matching.
        """
        mu_ = mod(arctan2(imag(m), real(m)), 2.0 * pi)
        kappa_ = VonMisesDistribution.besselratio_inverse(0, abs(m))
        vm = VonMisesDistribution(mu_, kappa_)
        return vm

    def __str__(self) -> str:
        return f"VonMisesDistribution: mu = {self.mu}, kappa = {self.kappa}"
