# pylint: disable=redefined-builtin,no-name-in-module,no-member
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
        assert kappa > 0, "kappa must be a positive scalar"
        assert (
            float(sqrt(rho1**2 + rho2**2)) < 1.0
        ), "sqrt(rho1^2 + rho2^2) must be strictly less than 1"

        self.mu = mu
        self.mu0 = mod(mu0, 2.0 * pi)
        self.kappa = kappa
        self.rho1 = rho1
        self.rho2 = rho2
        self.sigma = sigma

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
        assert xs.shape[-1] == 2

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
        assert n > 0, "n must be positive"
        s_vm = array(
            vonmises.rvs(kappa=float(self.kappa), loc=float(self.mu0), size=n)
            % (2.0 * float(pi))
        )
        muc, sigmac = self.get_mu_sigma(s_vm)
        s_gauss = array(norm.rvs(loc=muc, scale=float(sigmac)))
        return column_stack([s_vm, s_gauss])

    def linear_covariance(self, approximate_mean=None):
        """Return the intrinsic linear variance as a (1, 1) matrix.

        Returns:
            C (1, 1): [[sigma^2]]
        """
        _ = approximate_mean
        return array([[self.sigma**2]])

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
