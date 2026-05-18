import copy

import mpmath
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    allclose,
    array,
    concatenate,
    diag,
    exp,
    full,
    gammaln,
    hstack,
    linalg,
    log,
    ones,
    tile,
    zeros,
)

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .bingham_distribution import BinghamDistribution


class WatsonDistribution(AbstractHypersphericalDistribution):
    EPSILON = 1e-6

    def __init__(self, mu, kappa, norm_const: float | None = None):
        """
        Initializes a new instance of the WatsonDistribution class.

        Args:
            mu (): The mean direction of the distribution.
            kappa (float): The concentration parameter of the distribution.
        """
        AbstractHypersphericalDistribution.__init__(self, dim=mu.shape[0] - 1)
        assert mu.ndim == 1, "mu must be a 1-D vector"
        assert abs(linalg.norm(mu) - 1.0) < self.EPSILON, "mu is unnormalized"

        self.mu = mu
        self.kappa = kappa
        self._norm_const = norm_const
        self._ln_norm_const = log(norm_const) if norm_const is not None else None

    @property
    def norm_const(self):
        if self._norm_const is None:
            self._norm_const = array(
                float(
                    mpmath.gamma((self.dim + 1) / 2)
                    / (2 * float(pyrecest.backend.pi) ** ((self.dim + 1) / 2))
                    / mpmath.hyper([0.5], [(self.dim + 1) / 2.0], self.kappa)
                )
            )
        return self._norm_const

    @property
    def ln_norm_const(self):
        if self._ln_norm_const is None:
            self._ln_norm_const = array(
                (gammaln(array((self.dim + 1) / 2)))
                - log(2 * pyrecest.backend.pi ** ((self.dim + 1) / 2))
                - float(
                    mpmath.log(mpmath.hyper([0.5], [(self.dim + 1) / 2.0], self.kappa))
                )
            )
        return self._ln_norm_const

    def pdf(self, xs):
        """
        Computes the probability density function at xs.

        Args:
            xs: The values at which to evaluate the pdf.

        Returns:
            np.generic: The value of the pdf at xs.
        """
        assert xs.shape[-1] == self.input_dim, "Last dimension of xs must be dim + 1"
        p = self.norm_const * exp(self.kappa * (xs @ self.mu) ** 2)
        return p

    def ln_pdf(self, xs):
        return self.ln_norm_const + self.kappa * (xs @ self.mu) ** 2

    def to_bingham(self) -> BinghamDistribution:
        if self.kappa < 0:
            raise NotImplementedError(
                "Conversion to Bingham is not implemented for kappa<0"
            )

        M = tile(self.mu.reshape(-1, 1), (1, self.input_dim))
        E = diag(array(concatenate((array([0]), ones(self.input_dim - 1)))))
        M = M + E
        Q, _ = linalg.qr(M)
        M = hstack([Q[:, 1:], Q[:, 0].reshape(-1, 1)])
        Z = hstack((full((self.dim,), -self.kappa), array(0.0)))
        return BinghamDistribution(Z, M)

    def sample(self, n):
        if self.dim != 2:
            return self.to_bingham().sample(n)

        return super().sample(n)

    def mode(self):
        if self.kappa >= 0:
            return self.mu

        return self.mode_numerical()

    def set_mode(self, new_mode):
        assert new_mode.shape == self.mu.shape
        dist = copy.deepcopy(self)
        dist.mu = copy.deepcopy(new_mode)
        return dist

    def shift(self, shift_by):
        canonical_mu = concatenate((zeros(self.input_dim - 1), array([1.0])))
        assert allclose(self.mu, canonical_mu), (
            "There is no true shifting for the hypersphere. This is a function "
            "for compatibility and only works when mu is [0,0,...,1]."
        )
        return self.set_mode(shift_by)
