# pylint: disable=redefined-builtin,no-name-in-module,no-member
import copy
import math
from numbers import Integral

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
from scipy.optimize import brentq
from scipy.special import iv, ive
from scipy.stats import vonmises

from .abstract_circular_distribution import AbstractCircularDistribution


class VonMisesDistribution(AbstractCircularDistribution):
    """Von Mises distribution on the circle.

    References
    ----------
    Jammalamadaka, S. R., & SenGupta, A. (2001). Topics in Circular
    Statistics. World Scientific.
    """

    _MOMENT_MAGNITUDE_TOL = 1e-12

    def __init__(
        self,
        mu,
        kappa,
        norm_const: float | None = None,
    ):
        kappa_scalar = self._as_float_scalar(kappa, "kappa")
        if kappa_scalar < 0.0:
            raise ValueError("kappa must be nonnegative.")
        super().__init__()
        self.mu = mu
        self.kappa = kappa
        self._norm_const = norm_const
        self._norm_const_is_explicit = norm_const is not None

    def get_params(self):
        return self.mu, self.kappa

    @property
    def norm_const(self):
        if self._norm_const is None:
            self._norm_const = 2.0 * pi * iv(0, self.kappa)
        return self._norm_const

    def pdf(self, xs):
        xs = array(xs)
        if not getattr(self, "_norm_const_is_explicit", False):
            p = exp(self.kappa * (cos(xs - self.mu) - 1.0)) / (
                2.0 * pi * ive(0, self.kappa)
            )
        else:
            p = exp(self.kappa * cos(xs - self.mu)) / self.norm_const
        return p

    def sample(self, n):
        """Draw samples from the von Mises distribution."""
        if isinstance(n, bool) or not isinstance(n, Integral) or int(n) <= 0:
            raise ValueError("n must be a positive integer.")
        n = int(n)
        return mod(
            array(vonmises.rvs(kappa=float(self.kappa), loc=float(self.mu), size=n)),
            2.0 * pi,
        )

    def set_mean(self, mu):
        """Return a copy with a replaced mean direction.

        Parameters
        ----------
        mu : scalar
            New mean direction.
        """
        new_dist = copy.deepcopy(self)
        new_dist.mu = mu
        return new_dist

    @staticmethod
    def besselratio(nu, kappa):
        return ive(nu + 1, kappa) / ive(nu, kappa)

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
        xs = array(xs)
        if xs.ndim > 1:
            raise ValueError("xs must be a scalar or one-dimensional array")

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
    def _as_float_scalar(value, name: str) -> float:
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a scalar.") from exc

        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite.")
        return scalar

    @staticmethod
    def _besselratio_scalar(nu, kappa: float) -> float:
        if kappa == 0.0:
            return 0.0
        return float(ive(nu + 1, kappa) / ive(nu, kappa))

    @staticmethod
    def besselratio_inverse(v, x):
        x_scalar = VonMisesDistribution._as_float_scalar(x, "Bessel-ratio target")
        tol = VonMisesDistribution._MOMENT_MAGNITUDE_TOL

        if x_scalar < 0.0:
            if x_scalar >= -tol:
                return array(0.0)
            raise ValueError("Bessel-ratio target must lie in [0, 1).")
        if x_scalar <= tol:
            return array(0.0)
        if x_scalar >= 1.0 - tol:
            raise ValueError(
                "Bessel-ratio target must be strictly smaller than 1 for a finite "
                "von Mises concentration."
            )

        def residual(kappa: float) -> float:
            return VonMisesDistribution._besselratio_scalar(v, kappa) - x_scalar

        upper = 1.0
        while residual(upper) < 0.0:
            upper *= 2.0
            if upper > 1e12:
                raise ValueError(
                    "Bessel-ratio target is too close to 1 for stable finite inversion."
                )

        return array(brentq(residual, 0.0, upper, xtol=1e-14, rtol=1e-14, maxiter=100))

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
        result = (
            -self.kappa * VonMisesDistribution.besselratio(0, self.kappa)
            + self.kappa
            + log(array(2.0 * pi * ive(0, self.kappa)))
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
        kappa_ = VonMisesDistribution.besselratio_inverse(0, abs(m))
        if VonMisesDistribution._as_float_scalar(kappa_, "kappa") == 0.0:
            mu_ = array(0.0)
        else:
            mu_ = mod(arctan2(imag(m), real(m)), 2.0 * pi)
        vm = VonMisesDistribution(mu_, kappa_)
        return vm

    def __str__(self) -> str:
        return f"VonMisesDistribution: mu = {self.mu}, kappa = {self.kappa}"
