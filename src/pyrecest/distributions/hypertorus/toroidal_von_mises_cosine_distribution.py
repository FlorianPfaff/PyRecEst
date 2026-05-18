# pylint: disable=redefined-builtin,no-name-in-module,no-member
import math

import numpy as np
from pyrecest.backend import cos, mod, pi
from scipy.special import iv

from .abstract_toroidal_bivar_vm_distribution import AbstractToroidalBivarVMDistribution

_SERIES_RTOL = 1e-14
_SERIES_MIN_TERMS = 10
_SERIES_MAX_TERMS = 10000


def _iv(order, concentration):
    return float(iv(order, float(concentration)))


def _adaptive_series_sum(term, coefficient):
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
            1.0, abs(total)
        ):
            return math.fsum(terms)
    raise RuntimeError("Bivariate von Mises series did not converge")


def _symmetric_series_sum(term):
    return _adaptive_series_sum(term, lambda order: 1.0 if order == 0 else 2.0)


def _half_zero_series_sum(term):
    return _adaptive_series_sum(term, lambda order: 0.5 if order == 0 else 1.0)


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
        assert kappa3.shape == ()
        self.kappa3 = kappa3
        self.C = 1.0 / self.norm_const

    @property
    def norm_const(self):
        kappa0 = float(self.kappa[0])
        kappa1 = float(self.kappa[1])
        kappa3 = float(self.kappa3)

        def s(order):
            return _iv(order, kappa0) * _iv(order, kappa1) * _iv(order, -kappa3)

        return 4.0 * math.pi**2 * _symmetric_series_sum(s)

    def _coupling_term(self, xs):
        return -self.kappa3 * cos(xs[..., 0] - self.mu[0] - xs[..., 1] + self.mu[1])

    def trigonometric_moment(self, n):
        if n == 1:
            kappa0 = float(self.kappa[0])
            kappa1 = float(self.kappa[1])
            kappa3 = float(self.kappa3)

            def s1(order):
                return (
                    (_iv(order + 1, kappa0) + _iv(order - 1, kappa0))
                    * _iv(order, kappa1)
                    * _iv(order, -kappa3)
                )

            def s2(order):
                return (
                    _iv(order, kappa0)
                    * (_iv(order + 1, kappa1) + _iv(order - 1, kappa1))
                    * _iv(order, -kappa3)
                )

            def s(order):
                return _iv(order, kappa0) * _iv(order, kappa1) * _iv(order, -kappa3)

            s1_sum = _half_zero_series_sum(s1)
            s2_sum = _half_zero_series_sum(s2)
            s_sum = _symmetric_series_sum(s)

            # Use numpy directly here because the result is inherently complex
            # and pyrecest.backend does not support complex-valued arrays.
            m1 = s1_sum / s_sum * np.exp(1j * n * float(self.mu[0]))
            m2 = s2_sum / s_sum * np.exp(1j * n * float(self.mu[1]))
            return np.array([m1, m2])
        return self.trigonometric_moment_numerical(n)

    def shift(self, shift_by):
        assert shift_by.shape == (self.dim,)
        tvm = ToroidalVonMisesCosineDistribution(
            mod(self.mu + shift_by, 2.0 * pi), self.kappa, self.kappa3
        )
        return tvm
