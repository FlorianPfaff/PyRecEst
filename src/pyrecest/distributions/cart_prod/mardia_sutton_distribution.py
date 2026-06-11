# pylint: disable=redefined-builtin,no-name-in-module,no-member
import math

import numpy as np
from pyrecest.backend import (
    array,
    atleast_2d,
    column_stack,
    cos,
    exp,
    mod,
    pi,
    sin,
    sqrt,
)
from scipy.special import iv
from scipy.stats import norm, vonmises

from ..circle.von_mises_distribution import VonMisesDistribution
from .abstract_hypercylindrical_distribution import AbstractHypercylindricalDistribution


def _validate_positive_sample_count(n) -> int:
    count_array = np.asarray(n)
    if count_array.ndim != 0:
        raise ValueError("n must be a scalar integer")

    count = count_array.item()
    if isinstance(count, (bool, np.bool_)):
        raise ValueError("n must be an integer, not a boolean")

    try:
        count_int = int(count)
        count_float = float(count)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("n must be an integer") from exc

    if not np.isfinite(count_float) or not count_float.is_integer():
        raise ValueError("n must be a finite integer")
    if count_int <= 0:
        raise ValueError("n must be positive")
    return count_int


def _validate_positive_finite_scalar(value, name: str) -> float:
    scalar_array = np.asarray(value)
    if scalar_array.shape != ():
        raise ValueError(f"{name} must be a scalar")

    scalar = scalar_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite scalar")

    try:
        scalar_float = float(scalar)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite scalar") from exc

    if not math.isfinite(scalar_float) or scalar_float <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return scalar_float


class MardiaSuttonDistribution(AbstractHypercylindricalDistribution):
    """Gauss-von Mises distribution for cylindrical data (1 circular + 1 linear dimension).

    Mardia, K. V. & Sutton, T. W.
    A Model for Cylindrical Variables with Applications
    Journal of the Royal Statistical Society. Series B (Methodological),
    Wiley for the Royal Statistical Society, 1978, 40, pp. 229-233
    """

    def __init__(self, mu, mu0, kappa, rho1, rho2, sigma):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        """
        Parameters:
            mu (scalar): linear mean
            mu0 (scalar): circular mean (wrapped to [0, 2π))
            kappa (positive scalar): circular concentration
            rho1 (scalar): first correlation parameter
            rho2 (scalar): second correlation parameter
            sigma (positive scalar): linear standard deviation
        """
        AbstractHypercylindricalDistribution.__init__(self, bound_dim=1, lin_dim=1)
        kappa_float = _validate_positive_finite_scalar(kappa, "kappa")
        sigma_float = _validate_positive_finite_scalar(sigma, "sigma")
        rho_norm = float(sqrt(rho1**2 + rho2**2))
        if not math.isfinite(rho_norm) or rho_norm >= 1.0:
            raise ValueError("sqrt(rho1^2 + rho2^2) must be strictly less than 1")

        self.mu = mu
        self.mu0 = mod(mu0, 2.0 * pi)
        self.kappa = kappa_float
        self.rho1 = rho1
        self.rho2 = rho2
        self.sigma = sigma_float

    def get_mu_sigma(self, xa_circular):
        """Compute the conditional mean and std of the linear variable given circular values.

        Parameters:
            xa_circular: circular variable values

        Returns:
            muc: conditional mean of the linear variable (same shape as xa_circular)
            sigmac: conditional std of the linear variable (positive scalar)
        """
        muc = self.mu + self.sigma * sqrt(self.kappa) * (
            self.rho1 * (cos(xa_circular) - cos(self.mu0))
            + self.rho2 * (sin(xa_circular) - sin(self.mu0))
        )
        rho = sqrt(self.rho1**2 + self.rho2**2)
        sigmac = self.sigma * sqrt(1.0 - rho**2)
        return muc, sigmac

    def pdf(self, xs):
        """Evaluate the pdf at each row of xs.

        Parameters:
            xs (..., 2): locations where to evaluate the pdf;
                         first column is circular (θ), second is linear (x)

        Returns:
            p (...,): value of the pdf at each location
        """
        xs = atleast_2d(xs)
        if xs.shape[-1] != 2:
            raise ValueError("xs must contain circular and linear coordinates.")

        circular = xs[..., 0]
        linear = xs[..., 1]

        muc, sigmac = self.get_mu_sigma(circular)

        vm_part = exp(self.kappa * cos(circular - self.mu0)) / (
            2.0 * pi * iv(0, float(self.kappa))
        )
        gaussian_part = array(norm.pdf(linear, loc=muc, scale=float(sigmac)))

        return vm_part * gaussian_part

    def mode(self):
        """Return the mode of the distribution.

        Returns:
            m (2,): mode [mu0, mu] (circular first, then linear)
        """
        return array([self.mu0, self.mu])

    def sample(self, n):
        """Obtain n samples from the distribution.

        Parameters:
            n (int): number of samples

        Returns:
            s (n, 2): n samples on [0, 2π) × R (circular first, then linear)
        """
        n = _validate_positive_sample_count(n)
        s_vm = array(
            vonmises.rvs(kappa=float(self.kappa), loc=float(self.mu0), size=n)
            % (2.0 * float(pi))
        )
        muc, sigmac = self.get_mu_sigma(s_vm)
        s_gauss = array(norm.rvs(loc=muc, scale=float(sigmac)))
        return column_stack([s_vm, s_gauss])

    def linear_covariance(self, approximate_mean=None):
        """Return the marginal linear variance as a (1, 1) matrix.

        Returns:
            C (1, 1): marginal variance of the linear variable
        """
        _ = approximate_mean

        kappa = float(self.kappa)
        bessel_0 = iv(0, kappa)
        bessel_ratio_1 = iv(1, kappa) / bessel_0
        bessel_ratio_2 = iv(2, kappa) / bessel_0

        rho_squared = self.rho1**2 + self.rho2**2
        conditional_variance = self.sigma**2 * (1.0 - rho_squared)

        # The conditional linear mean depends on the von-Mises-distributed angle.
        # Hence the marginal linear variance is E[Var[X | theta]] plus
        # Var(E[X | theta]); returning sigma**2 alone ignores the second term.
        aligned_rho_cos = self.rho1 * cos(self.mu0) + self.rho2 * sin(self.mu0)
        aligned_rho_sin = -self.rho1 * sin(self.mu0) + self.rho2 * cos(self.mu0)
        cos_variance = 0.5 * (1.0 + bessel_ratio_2) - bessel_ratio_1**2
        sin_variance = 0.5 * (1.0 - bessel_ratio_2)
        conditional_mean_variance = (
            self.sigma**2
            * self.kappa
            * (aligned_rho_cos**2 * cos_variance + aligned_rho_sin**2 * sin_variance)
        )

        return array([[conditional_variance + conditional_mean_variance]])

    def marginalize_linear(self):
        """Return the marginal circular distribution.

        The marginal over the linear variable is a Von Mises distribution
        since the conditional Gaussian integrates to one.

        Returns:
            vm: VonMisesDistribution(mu0, kappa)
        """
        return VonMisesDistribution(self.mu0, self.kappa)

    def marginalize_periodic(self):
        """Return the marginal linear distribution by integrating out the circular variable.

        Returns:
            dist: CustomLinearDistribution representing the marginal over the linear variable
        """
        from scipy.integrate import quad  # pylint: disable=import-outside-toplevel

        from ..nonperiodic.custom_linear_distribution import (  # pylint: disable=import-outside-toplevel
            CustomLinearDistribution,
        )

        def marginal_pdf(xs):
            results = []
            for x in xs.ravel():
                val, _ = quad(
                    lambda theta, x_=x: float(self.pdf(array([[theta, float(x_)]]))[0]),
                    0.0,
                    2.0 * float(pi),
                )
                results.append(val)
            return array(results)

        return CustomLinearDistribution(marginal_pdf, 1)

    def get_reasonable_integration_boundaries(self, scalingFactor=10):
        sigma = float(self.sigma)
        mu_lin = float(self.mu)
        return [
            [0.0, 2.0 * float(pi)],
            [mu_lin - scalingFactor * sigma, mu_lin + scalingFactor * sigma],
        ]

    def integrate_numerically(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.get_reasonable_integration_boundaries()

        from scipy.integrate import nquad  # pylint: disable=import-outside-toplevel

        def f(theta, x):
            return float(self.pdf(array([[theta, x]]))[0])

        return nquad(f, integration_boundaries)[0]
