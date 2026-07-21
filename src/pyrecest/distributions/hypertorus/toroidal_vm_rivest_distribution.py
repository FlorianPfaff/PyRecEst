# pylint: disable=redefined-builtin,no-name-in-module,no-member
import math

from pyrecest.backend import array, cos, exp, mod, pi, sin, sqrt
from scipy.special import ive

from ._input_validation import as_shift_vector
from .abstract_toroidal_bivar_vm_distribution import (
    validate_scalar_parameter,
    validate_toroidal_vm_parameters,
)
from .abstract_toroidal_distribution import AbstractToroidalDistribution


class ToroidalVMRivestDistribution(AbstractToroidalDistribution):
    """
    Bivariate von Mises distribution (Rivest version) with two correlation
    parameters alpha and beta, corresponding to A = diag([alpha, beta]).

    Rivest, L.-P.
    A Distribution for Dependent Unit Vectors
    Communications in Statistics - Theory and Methods, 1988, 17, 461-483
    """

    def __init__(self, mu, kappa, alpha, beta):
        AbstractToroidalDistribution.__init__(self)
        mu, kappa = validate_toroidal_vm_parameters(mu, kappa)
        alpha = validate_scalar_parameter(alpha, "alpha")
        beta = validate_scalar_parameter(beta, "beta")

        self.mu = mod(mu, 2.0 * pi)
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta

        self._bessel_exponential_scale = self._series_exponential_scale()
        self._scaled_norm_series_sum = self._compute_scaled_norm_series_sum()
        self._scaled_norm_const = 4.0 * math.pi**2 * self._scaled_norm_series_sum
        if not math.isfinite(self._scaled_norm_const) or self._scaled_norm_const <= 0.0:
            raise FloatingPointError(
                "Rivest normalization series must be positive and finite"
            )
        self._log_norm_const = (
            math.log(self._scaled_norm_const) + self._bessel_exponential_scale
        )
        self.C = math.exp(-self._log_norm_const)

    def _series_arguments(self):
        return (
            float(self.kappa[0]),
            float(self.kappa[1]),
            float((self.alpha + self.beta) / 2.0),
            float((self.alpha - self.beta) / 2.0),
        )

    def _series_exponential_scale(self):
        return sum(abs(argument) for argument in self._series_arguments())

    def _compute_scaled_norm_series_sum(self):
        kappa0, kappa1, alpha_plus, alpha_minus = self._series_arguments()
        terms = []
        n = 10
        for j in range(-n, n + 1):
            for ell in range(-n, n + 1):
                if (j + ell) % 2 == 0:
                    terms.append(
                        float(ive(j, kappa0))
                        * float(ive(ell, kappa1))
                        * float(ive((j + ell) // 2, alpha_plus))
                        * float(ive((j - ell) // 2, alpha_minus))
                    )
        return math.fsum(terms)

    @property
    def log_norm_const(self):
        """Return the logarithm of the normalization constant."""
        return self._log_norm_const

    @property
    def norm_const(self):
        try:
            return math.exp(self.log_norm_const)
        except OverflowError:
            return float("inf")

    def pdf(self, xs):
        xs = array(xs)
        if xs.ndim == 0 or xs.shape[-1] != self.dim:
            raise ValueError(
                f"xs must have trailing dimension {self.dim}, got {xs.shape}."
            )
        log_unnormalized = (
            self.kappa[0] * cos(xs[..., 0] - self.mu[0])
            + self.kappa[1] * cos(xs[..., 1] - self.mu[1])
            + self.alpha * cos(xs[..., 0] - self.mu[0]) * cos(xs[..., 1] - self.mu[1])
            + self.beta * sin(xs[..., 0] - self.mu[0]) * sin(xs[..., 1] - self.mu[1])
        )
        return exp(log_unnormalized - self._bessel_exponential_scale) / (
            self._scaled_norm_const
        )

    def trigonometric_moment(self, n):
        if n == 1:
            kappa0, kappa1, alpha_plus, alpha_minus = self._series_arguments()
            terms1 = []
            terms2 = []
            m = 10
            for j in range(-m, m + 1):
                for ell in range(-m, m + 1):
                    if (j + ell) % 2 == 0:
                        bessel_jl = float(ive((j + ell) // 2, alpha_plus)) * float(
                            ive((j - ell) // 2, alpha_minus)
                        )
                        terms1.append(
                            (float(ive(j + 1, kappa0)) + float(ive(j - 1, kappa0)))
                            * float(ive(ell, kappa1))
                            * bessel_jl
                        )
                        terms2.append(
                            float(ive(j, kappa0))
                            * (
                                float(ive(ell + 1, kappa1))
                                + float(ive(ell - 1, kappa1))
                            )
                            * bessel_jl
                        )
            total1 = math.fsum(terms1)
            total2 = math.fsum(terms2)
            m1 = (
                total1
                / (2.0 * self._scaled_norm_series_sum)
                * exp(1j * float(self.mu[0]))
            )
            m2 = (
                total2
                / (2.0 * self._scaled_norm_series_sum)
                * exp(1j * float(self.mu[1]))
            )
            return array([m1, m2])
        return self.trigonometric_moment_numerical(n)

    def _correlation_series_sums(self):
        """Compute scaled double-series sums needed for circular correlation."""
        kappa0, kappa1, alpha_plus, alpha_minus = self._series_arguments()
        terms0 = []
        terms1 = []
        terms2 = []
        m = 10
        for j in range(-m, m + 1):
            for ell in range(-m, m + 1):
                if (j + ell) % 2 == 0:
                    jl_half = (j + ell) // 2
                    jl_diff = (j - ell) // 2
                    iv_ab = float(ive(jl_half, alpha_plus)) * float(
                        ive(jl_diff, alpha_minus)
                    )
                    terms0.append(
                        float(ive(j, kappa0))
                        * float(ive(ell, kappa1))
                        * (
                            (
                                float(ive(jl_half + 1, alpha_plus))
                                + float(ive(jl_half - 1, alpha_plus))
                            )
                            * float(ive(jl_diff, alpha_minus))
                            - float(ive(jl_half, alpha_plus))
                            * (
                                float(ive(jl_diff + 1, alpha_minus))
                                + float(ive(jl_diff - 1, alpha_minus))
                            )
                        )
                    )
                    terms1.append(
                        (
                            float(ive(j + 2, kappa0)) / 2.0
                            + float(ive(j, kappa0))
                            + float(ive(j - 2, kappa0)) / 2.0
                        )
                        * float(ive(ell, kappa1))
                        * iv_ab
                    )
                    terms2.append(
                        float(ive(j, kappa0))
                        * (
                            float(ive(ell + 2, kappa1)) / 2.0
                            + float(ive(ell, kappa1))
                            + float(ive(ell - 2, kappa1)) / 2.0
                        )
                        * iv_ab
                    )
        return math.fsum(terms0), math.fsum(terms1), math.fsum(terms2)

    def circular_correlation_jammalamadaka(self):
        total0, total1, total2 = self._correlation_series_sums()
        e_sin_a_sin_b = total0 / (4.0 * self._scaled_norm_series_sum)
        e_sin_a_sq = 1.0 - total1 / (2.0 * self._scaled_norm_series_sum)
        e_sin_b_sq = 1.0 - total2 / (2.0 * self._scaled_norm_series_sum)
        return e_sin_a_sin_b / sqrt(e_sin_a_sq * e_sin_b_sq)

    def shift(self, shift_by):
        shift_by = as_shift_vector(shift_by, self.dim)
        return ToroidalVMRivestDistribution(
            mod(self.mu + shift_by, 2.0 * pi),
            self.kappa,
            self.alpha,
            self.beta,
        )
