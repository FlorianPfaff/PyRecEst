# pylint: disable=redefined-builtin,no-name-in-module,no-member
import numpy as np
from pyrecest.backend import array, cos, mod, pi, sum
from scipy.special import iv

from .abstract_toroidal_bivar_vm_distribution import AbstractToroidalBivarVMDistribution

_SERIES_TERMS = 10


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
        def s(p):
            return iv(p, self.kappa[0]) * iv(p, self.kappa[1]) * iv(p, -self.kappa3)

        Cinv = (
            4.0
            * pi**2
            * (s(0) + 2.0 * sum(array([s(p) for p in range(1, _SERIES_TERMS + 1)])))
        )
        return Cinv

    def _coupling_term(self, xs):
        return -self.kappa3 * cos(xs[..., 0] - self.mu[0] - xs[..., 1] + self.mu[1])

    def trigonometric_moment(self, n):
        if n == 1:

            def s1(m):
                return (
                    (iv(m + 1, self.kappa[0]) + iv(m - 1, self.kappa[0]))
                    * iv(m, self.kappa[1])
                    * iv(m, -self.kappa3)
                )

            def s2(m):
                return (
                    iv(m, self.kappa[0])
                    * (iv(m + 1, self.kappa[1]) + iv(m - 1, self.kappa[1]))
                    * iv(m, -self.kappa3)
                )

            def s(p):
                return iv(p, self.kappa[0]) * iv(p, self.kappa[1]) * iv(p, -self.kappa3)

            terms = range(1, _SERIES_TERMS + 1)
            s1_sum = s1(0) / 2.0 + sum(array([s1(m) for m in terms]))
            s2_sum = s2(0) / 2.0 + sum(array([s2(m) for m in terms]))
            s_sum = s(0) + 2.0 * sum(array([s(p) for p in terms]))

            # Use numpy directly here because the result is inherently complex
            # and pyrecest.backend does not support complex-valued arrays.
            m1 = float(s1_sum) / float(s_sum) * np.exp(1j * n * float(self.mu[0]))
            m2 = float(s2_sum) / float(s_sum) * np.exp(1j * n * float(self.mu[1]))
            return np.array([m1, m2])
        return self.trigonometric_moment_numerical(n)

    def shift(self, shift_by):
        assert shift_by.shape == (self.dim,)
        tvm = ToroidalVonMisesCosineDistribution(
            mod(self.mu + shift_by, 2.0 * pi), self.kappa, self.kappa3
        )
        return tvm
