# pylint: disable=redefined-builtin,no-name-in-module,no-member
import math

import numpy as np
from pyrecest.backend import cos, mod, pi
from scipy.special import iv, ive

from ._input_validation import as_shift_vector
from .abstract_toroidal_bivar_vm_distribution import (
    AbstractToroidalBivarVMDistribution,
    validate_scalar_parameter,
)

_SERIES_RTOL = 1e-14
_SERIES_MIN_TERMS = 10
_SERIES_MAX_TERMS = 10000


def _iv(order, concentration):
    return float(iv(order, float(concentration)))


def _scaled_iv(order, concentration):
    return float(ive(order, float(concentration)))


def _adaptive_series_sum(term, coefficient, *, scale_floor=1.0):
    """Sum a scalar series until the latest contribution is negligible."""
    total = 0.0
    terms = []
    for order in range(_SERIES_MAX_TERMS + 1):
        contribution = float(coefficient(order) * term(order))
        if not math.isfinite(contribution):
            raise FloatingPointError("Bivariate von Mises series term is not finite")
        terms.append(contribution)
        total += contribution
        if order >= _SERIES_MIN_TERMS and abs(contribution) <= _SERIES_RTOL * max(
            scale_floor, abs(total)
        ):
            return math.fsum(terms)
    raise RuntimeError("Bivariate von Mises series did not converge")


def _symmetric_series_sum(term, *, scale_floor=1.0):
    return _adaptive_series_sum(
        term,
        lambda order: 1.0 if order == 0 else 2.0,
        scale_floor=scale_floor,
    )


def _half_zero_series_sum(term, *, scale_floor=1.0):
    return _adaptive_series_sum(
        term,
        lambda order: 0.5 if order == 0 else 1.0,
        scale_floor=scale_floor,
    )


class ToroidalVonMisesCosineDistribution(AbstractToroidalBivarVMDistribution):
    """Bivariate von Mises distribution, cosine model.

    Corresponds to A = [-kappa3, 0; 0, -kappa3].

    References:
        Mardia, K. V.; Taylor, C. C. & Subramaniam, G. K.
        Protein Bioinformatics and Mixtures of Bivariate von Mises Distributions
        for Angular Data Biometrics, 2007, 63, 505-512

        Mardia, K. V. & Frellsen, J. in Hamelryck, T.; Mardia, K. &
        Ferkinghoff-Borg, J. (Eds.)
        Statistics of Bivariate von Mises Distributions
        Bayesian Methods in Structural Bioinformatics,
        Springer Berlin Heidelberg, 2012, 159-178
    """

    def __init__(self, mu, kappa, kappa3):
        AbstractToroidalBivarVMDistribution.__init__(self, mu, kappa)
        kappa3 = validate_scalar_parameter(kappa3, "kappa3")
        self.kappa3 = kappa3
        self._bessel_exponential_scale = (
            float(self.kappa[0]) + float(self.kappa[1]) + abs(float(self.kappa3))
        )

        try:
            self._norm_series_sum = self._compute_norm_series_sum(scaled=False)
            self._norm_const = 4.0 * math.pi**2 * self._norm_series_sum
            if not math.isfinite(self._norm_const) or self._norm_const <= 0.0:
                raise FloatingPointError
            self._series_uses_scaled_bessel = False
            self._log_norm_const = math.log(self._norm_const)
        except (FloatingPointError, OverflowError):
            self._norm_series_sum = self._compute_norm_series_sum(scaled=True)
            scaled_norm_const = 4.0 * math.pi**2 * self._norm_series_sum
            if not math.isfinite(scaled_norm_const) or scaled_norm_const <= 0.0:
                raise FloatingPointError(
                    "Bivariate von Mises normalization constant must be "
                    "positive and finite"
                )
            self._series_uses_scaled_bessel = True
            self._log_norm_const = (
                math.log(scaled_norm_const) + self._bessel_exponential_scale
            )
            try:
                self._norm_const = math.exp(self._log_norm_const)
            except OverflowError:
                self._norm_const = float("inf")

        self.C = math.exp(-self._log_norm_const)

    def _compute_norm_series_sum(self, *, scaled):
        kappa0 = float(self.kappa[0])
        kappa1 = float(self.kappa[1])
        kappa3 = float(self.kappa3)
        bessel = _scaled_iv if scaled else _iv

        def s(order):
            return (
                bessel(order, kappa0) * bessel(order, kappa1) * bessel(order, -kappa3)
            )

        scale_floor = 0.0 if scaled else 1.0
        return _symmetric_series_sum(s, scale_floor=scale_floor)

    @property
    def log_norm_const(self):
        """Return the logarithm of the normalization constant."""
        return self._log_norm_const

    @property
    def norm_const(self):
        return self._norm_const

    def _coupling_term(self, xs):
        return -self.kappa3 * cos(xs[..., 0] - self.mu[0] - xs[..., 1] + self.mu[1])

    def trigonometric_moment(self, n):
        if n == 1:
            kappa0 = float(self.kappa[0])
            kappa1 = float(self.kappa[1])
            kappa3 = float(self.kappa3)
            bessel = _scaled_iv if self._series_uses_scaled_bessel else _iv
            scale_floor = 0.0 if self._series_uses_scaled_bessel else 1.0

            def s1(order):
                return (
                    (bessel(order + 1, kappa0) + bessel(order - 1, kappa0))
                    * bessel(order, kappa1)
                    * bessel(order, -kappa3)
                )

            def s2(order):
                return (
                    bessel(order, kappa0)
                    * (bessel(order + 1, kappa1) + bessel(order - 1, kappa1))
                    * bessel(order, -kappa3)
                )

            s1_sum = _half_zero_series_sum(s1, scale_floor=scale_floor)
            s2_sum = _half_zero_series_sum(s2, scale_floor=scale_floor)
            s_sum = self._norm_series_sum

            # Use numpy directly here because the result is inherently complex
            # and pyrecest.backend does not support complex-valued arrays.
            m1 = s1_sum / s_sum * np.exp(1j * n * float(self.mu[0]))
            m2 = s2_sum / s_sum * np.exp(1j * n * float(self.mu[1]))
            return np.array([m1, m2])
        return self.trigonometric_moment_numerical(n)

    def shift(self, shift_by):
        shift_by = as_shift_vector(shift_by, self.dim)
        tvm = ToroidalVonMisesCosineDistribution(
            mod(self.mu + shift_by, 2.0 * pi), self.kappa, self.kappa3
        )
        return tvm
