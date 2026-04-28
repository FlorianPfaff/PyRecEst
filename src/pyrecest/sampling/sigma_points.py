"""
Sigma-point sampling schemes for unscented transforms.
"""

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import asarray
from pyrecest.backend import copy as backend_copy
from pyrecest.backend import empty, float64, full, linalg, reshape


class MerweScaledSigmaPoints:
    """Merwe scaled sigma points (van der Merwe, 2004).

    Parameters
    ----------
    n:
        State dimension.
    alpha:
        Spread of sigma points around the mean (typically 1e-3).
    beta:
        Prior knowledge of the distribution (2 is optimal for Gaussians).
    kappa:
        Secondary scaling parameter (typically 0).
    """

    def __init__(self, n: int, alpha: float, beta: float, kappa: float):
        assert pyrecest.backend.__backend_name__ not in (
            "jax",
        ), "MerweScaledSigmaPoints is not supported on the JAX backend"
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._compute_weights()

    def _compute_weights(self):
        n = self.n
        lam = self.alpha**2 * (n + self.kappa) - n

        self.Wm = full(2 * n + 1, 0.5 / (n + lam))
        self.Wm[0] = lam / (n + lam)

        self.Wc = backend_copy(self.Wm)
        self.Wc[0] = lam / (n + lam) + (1.0 - self.alpha**2 + self.beta)

    def sigma_points(self, x, P):
        """Return ``(2n+1, n)`` sigma-point matrix.

        Parameters
        ----------
        x:
            State mean, shape ``(n,)``.
        P:
            State covariance, shape ``(n, n)``.
        """
        n = self.n
        lam = self.alpha**2 * (n + self.kappa) - n

        x = reshape(asarray(x, dtype=float64), (-1,))
        P = asarray(P, dtype=float64)

        U = linalg.cholesky((n + lam) * P)  # lower-triangular

        sigmas = empty((2 * n + 1, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1] = x + U[:, i]
            sigmas[n + i + 1] = x - U[:, i]
        return sigmas


class JulierSigmaPoints:
    """Julier sigma points (Julier and Uhlmann, 1997).

    Parameters
    ----------
    n:
        State dimension.
    kappa:
        Scaling parameter (``n + kappa`` should be non-zero).
    """

    def __init__(self, n: int, kappa: float = 0.0):
        assert pyrecest.backend.__backend_name__ not in (
            "jax",
        ), "JulierSigmaPoints is not supported on the JAX backend"
        self.n = n
        self.kappa = kappa
        self._compute_weights()

    def _compute_weights(self):
        n = self.n
        k = n + self.kappa

        self.Wm = full(2 * n + 1, 0.5 / k)
        self.Wm[0] = self.kappa / k

        self.Wc = backend_copy(self.Wm)

    def sigma_points(self, x, P):
        """Return ``(2n+1, n)`` sigma-point matrix.

        Parameters
        ----------
        x:
            State mean, shape ``(n,)``.
        P:
            State covariance, shape ``(n, n)``.
        """
        n = self.n
        k = n + self.kappa

        x = reshape(asarray(x, dtype=float64), (-1,))
        P = asarray(P, dtype=float64)

        U = linalg.cholesky(k * P)  # lower-triangular

        sigmas = empty((2 * n + 1, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1] = x + U[:, i]
            sigmas[n + i + 1] = x - U[:, i]
        return sigmas
