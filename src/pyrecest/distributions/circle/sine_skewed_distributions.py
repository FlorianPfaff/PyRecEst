from abc import abstractmethod
from numbers import Integral

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos, cosh, exp, mod, ndim, pi, sin, sinh
from scipy.special import ive  # pylint: disable=no-name-in-module

from .abstract_circular_distribution import AbstractCircularDistribution
from .von_mises_distribution import VonMisesDistribution
from .wrapped_cauchy_distribution import WrappedCauchyDistribution
from .wrapped_normal_distribution import WrappedNormalDistribution


def _validate_sine_power(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_scalar_angle(value):
    angle = array(value)
    if ndim(angle) != 0:
        raise ValueError("angle must be a scalar")
    return angle


class GeneralizedKSineSkewedVonMisesDistribution(AbstractCircularDistribution):
    """
    See Bekker, A., Nakhaei Rad, N., Arashi, M., Ley, C. (2020). Generalized Skew-Symmetric Circular and
    Toroidal Distributions, Florence Nightingale Directional Statistics volume, Springer.
    Parameters:
    - mu (float): Mean direction of the distribution.
    - kappa (float): Concentration parameter (non-negative).
    - lambda_ (float): Skewness parameter, must be between -1 and 1 inclusive.
    - k (int): Sine multiplier, currently supports only k=1.
    - m (int): Power of the sine term, must be a positive integer.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(self, mu, kappa, lambda_, k, m):
        AbstractCircularDistribution.__init__(self)
        self.mu = mod(mu, 2 * pi)
        self.kappa = kappa
        self.lambda_ = lambda_
        self.k = k
        self.m = m

        self.validate_parameters()

    def validate_parameters(self):
        if not (-1.0 <= self.lambda_ <= 1.0):
            raise ValueError("lambda_ must be between -1 and 1 inclusive")
        self.m = _validate_sine_power(self.m, "m")

    def pdf(self, xs):
        # Evaluate the von Mises distribution and multiply by (1 + lambda_ * sin(xa - mu))
        if self.k != 1:
            raise NotImplementedError("Currently, only k=1 is supported")
        vm_pdf = VonMisesDistribution(self.mu, self.kappa).pdf(xs)
        skew_factor = (1 + self.lambda_ * sin(self.k * (xs - self.mu))) ** self.m
        if self.m == 1:
            norm_const = 1
        elif self.m == 2:
            norm_const = 1 / (
                1 + self.lambda_**2 / 2 * (1 - bessel_ratio(2, self.kappa))
            )
        elif self.m == 3:
            norm_const = 1 / (
                1 + 3 * self.lambda_**2 / 2 * (1 - bessel_ratio(2, self.kappa))
            )
        elif self.m == 4:
            norm_const = 1 / (
                1
                + self.lambda_**4
                / 8
                * (3 - 4 * bessel_ratio(2, self.kappa) + bessel_ratio(4, self.kappa))
                + 3 * self.lambda_**2 * (1 - bessel_ratio(2, self.kappa))
            )
        else:
            raise NotImplementedError("m > 4 not implemented")

        return norm_const * vm_pdf * skew_factor

    def shift(self, shift_by):
        shift_by = _validate_scalar_angle(shift_by)
        new_dist = GeneralizedKSineSkewedVonMisesDistribution(
            self.mu + shift_by, self.kappa, self.lambda_, self.k, self.m
        )
        return new_dist


class SineSkewedVonMisesDistribution(GeneralizedKSineSkewedVonMisesDistribution):
    def __init__(self, mu, kappa, lambda_):
        super().__init__(mu, kappa, lambda_, k=1, m=1)


class GSSVMDistribution(GeneralizedKSineSkewedVonMisesDistribution):
    """
    Generalized Skew-Symmetric Von Mises (GSSVM) distribution.

    Special case of GeneralizedKSineSkewedVonMisesDistribution with k=1 fixed.
    Corresponds to GSSVMDistribution in libDirectional.

    Parameters:
    - mu (float): Mean direction of the distribution.
    - kappa (float): Concentration parameter (non-negative).
    - lambda_ (float): Skewness parameter, must be between -1 and 1 inclusive.
    - n (int): Order/power of the sine skewing term, must be a positive integer.
    """

    def __init__(self, mu, kappa, lambda_, n):
        super().__init__(mu, kappa, lambda_, k=1, m=n)

    @property
    def n(self):
        return self.m

    def shift(self, shift_by):
        shift_by = _validate_scalar_angle(shift_by)
        return GSSVMDistribution(self.mu + shift_by, self.kappa, self.lambda_, self.n)


def bessel_ratio(p, z):
    """
    Computes the ratio I_p(z) / I_0(z) in a numerically stable manner using
    exponentially scaled modified Bessel functions.

    Parameters:
    - p: Order of the Bessel function.
    - z: Argument for the Bessel function.

    Returns:
    - The ratio I_p(z) / I_0(z), calculated in a numerically stable way.
    """
    # ive(p, z) = iv(p, z) * exp(-|z|), so ive(p, z) / ive(0, z) = iv(p, z) / iv(0, z).
    return ive(p, z) / ive(0, z)


class AbstractSineSkewedDistribution(AbstractCircularDistribution):
    """
    Abstract superclass for sine-skewed distributions.
    """

    def __init__(self, mu, lambda_):
        """
        Initialize the sine-skewed distribution with a central location parameter mu
        and a skewness parameter lambda_.
        """
        AbstractCircularDistribution.__init__(self)
        self.mu = mu
        self.lambda_ = array(lambda_)
        self.validate_parameters()

    def validate_parameters(self):
        """Validate parameters common to first-order sine-skewed distributions."""
        if ndim(self.lambda_) != 0:
            raise ValueError("lambda_ must be a scalar")

        if not (-1.0 <= self.lambda_ <= 1.0):
            raise ValueError("lambda_ must be between -1 and 1 inclusive")

    @abstractmethod
    def base_pdf(self, xs):
        """
        Compute the base probability density function (PDF) for the wrapped distribution
        without skewness. This method must be implemented by subclasses.
        """

    def pdf(self, xs):
        """
        Compute the skewed probability density function (PDF) for the distribution.
        """
        # Calculate the base pdf from the wrapped distribution
        base_pdf = self.base_pdf(xs)

        # Apply the skewing factor
        skewed_pdf = base_pdf * (1 + self.lambda_ * sin(xs - self.mu))

        return skewed_pdf


class SineSkewedWrappedNormalDistribution(AbstractSineSkewedDistribution):
    def __init__(self, mu, sigma, lambda_):
        super().__init__(mu, lambda_)
        self.wrapped_normal = WrappedNormalDistribution(mu, sigma)

    @property
    def sigma(self):
        return self.wrapped_normal.sigma

    def base_pdf(self, xs):
        return self.wrapped_normal.pdf(xs)


class SineSkewedWrappedCauchyDistribution(AbstractSineSkewedDistribution):
    def __init__(self, mu, gamma, lambda_):
        super().__init__(mu, lambda_)
        self.wrapped_cauchy = WrappedCauchyDistribution(mu, gamma)

    @property
    def gamma(self):
        return self.wrapped_cauchy.gamma

    def base_pdf(self, xs):
        return self.wrapped_cauchy.pdf(xs)


class GeneralizedKSineSkewedWrappedCauchyDistribution(AbstractCircularDistribution):
    """
    Generalized K Sine-Skewed Wrapped Cauchy (GSSC) distribution.
    See Bekker, A., Nakhaei Rad, N., Arashi, M., Ley, C. (2020). Generalized Skew-Symmetric Circular and
    Toroidal Distributions, Florence Nightingale Directional Statistics volume, Springer.
    Parameters:
    - mu (float): Mean direction of the distribution.
    - gamma (float): Concentration parameter of the wrapped Cauchy distribution (positive).
    - lambda_ (float): Skewness parameter, must be between -1 and 1 inclusive.
    - k (int): Sine multiplier, currently supports only k=1.
    - m (int): Power of the sine term, must be a positive integer.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(self, mu, gamma, lambda_, k, m):
        AbstractCircularDistribution.__init__(self)
        self.mu = mod(mu, 2 * pi)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.k = k
        self.m = m

        self.validate_parameters()

    def validate_parameters(self):
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if not (-1.0 <= self.lambda_ <= 1.0):
            raise ValueError("lambda_ must be between -1 and 1 inclusive")
        self.m = _validate_sine_power(self.m, "m")

    def pdf(self, xs):
        if self.k != 1:
            raise NotImplementedError("Currently, only k=1 is supported")
        # Use the WC pdf formula directly to ensure correct centering at mu
        wc_pdf_vals = (
            1 / (2 * pi) * sinh(self.gamma) / (cosh(self.gamma) - cos(xs - self.mu))
        )
        skew_factor = (1 + self.lambda_ * sin(self.k * (xs - self.mu))) ** self.m
        # For the wrapped Cauchy: E[cos(n*(x-mu))] = exp(-n*k*gamma)
        r2 = exp(-2 * self.k * self.gamma)
        r4 = exp(-4 * self.k * self.gamma)
        if self.m == 1:
            norm_const = 1
        elif self.m == 2:
            norm_const = 1 / (1 + self.lambda_**2 / 2 * (1 - r2))
        elif self.m == 3:
            norm_const = 1 / (1 + 3 * self.lambda_**2 / 2 * (1 - r2))
        elif self.m == 4:
            norm_const = 1 / (
                1
                + self.lambda_**4 / 8 * (3 - 4 * r2 + r4)
                + 3 * self.lambda_**2 * (1 - r2)
            )
        else:
            raise NotImplementedError("m > 4 not implemented")

        return norm_const * wc_pdf_vals * skew_factor

    def shift(self, shift_by):
        shift_by = _validate_scalar_angle(shift_by)
        return GeneralizedKSineSkewedWrappedCauchyDistribution(
            self.mu + shift_by, self.gamma, self.lambda_, self.k, self.m
        )
