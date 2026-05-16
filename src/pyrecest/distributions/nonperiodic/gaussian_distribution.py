import copy
from numbers import Integral

# pylint: disable=no-name-in-module
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import linalg, matvec, ndim, random, reshape
from scipy.linalg import cholesky

from .abstract_linear_distribution import AbstractLinearDistribution


class GaussianDistribution(AbstractLinearDistribution):
    """Multivariate Gaussian distribution on a Euclidean space.

    Parameters
    ----------
    mu : backend array, shape (n,) or 0-D scalar array
        Mean vector. A 0-D backend array is treated as a one-dimensional
        Gaussian.
    C : backend array, shape (n, n) or 0-D scalar array
        Positive-definite covariance matrix. A 0-D backend array is treated as
        the covariance of a one-dimensional Gaussian.
    check_validity : bool, optional
        If true, validate that ``C`` is positive definite.

    Attributes
    ----------
    mu : array-like, shape (n,)
        Mean vector.
    C : array-like, shape (n, n)
        Covariance matrix.
    """

    def __init__(self, mu, C, check_validity=True):
        assert ndim(mu) <= 1, "mu must be 1-dimensional"
        if ndim(mu) == 0:
            mu = mu.reshape((1,))
            C = C.reshape((1, 1))
        assert ndim(C) == 2, "C must be 2-dimensional"
        AbstractLinearDistribution.__init__(self, dim=mu.shape[0])
        assert (
            1 == mu.shape[0] == C.shape[0] or mu.shape[0] == C.shape[0] == C.shape[1]
        ), "Size of C invalid"
        self.mu = mu

        if check_validity:
            if self.dim == 1:
                assert C > 0, "C must be positive definite"
            elif self.dim == 2:
                assert (
                    C[0, 0] > 0.0 and linalg.det(C) > 0.0
                ), "C must be positive definite"
            else:
                cholesky(C)  # Will fail if C is not positive definite

        self.C = C

    def set_mean(self, new_mean):
        """Return a copy with a replaced mean vector.

        Parameters
        ----------
        new_mean : array-like, shape (n,)
            New mean vector.
        """
        new_dist = copy.deepcopy(self)
        new_dist.mu = new_mean
        return new_dist

    def pdf(self, xs):
        """Evaluate the probability density.

        Parameters
        ----------
        xs : array-like, shape (n,) or (..., n)
            Evaluation point or batch of evaluation points. For a
            one-dimensional Gaussian, one-dimensional arrays are interpreted as
            a batch of scalar evaluation points.

        Returns
        -------
        array-like
            Density values with one value per evaluation point.
        """
        assert (
            self.dim == 1 and xs.ndim <= 1 or xs.shape[-1] == self.dim
        ), "Dimension incorrect"
        if pyrecest.backend.__backend_name__ == "numpy":
            from scipy.stats import multivariate_normal as mvn

            pdfvals = mvn.pdf(xs, self.mu, self.C)
        elif pyrecest.backend.__backend_name__ == "pytorch":
            # Disable import errors for megalinter
            import torch as _torch  # pylint: disable=import-error

            distribution = _torch.distributions.MultivariateNormal(self.mu, self.C)
            if xs.ndim == 1 and self.dim == 1:
                # For 1-D distributions, we need to reshape the input to a 2-D tensor
                # to be able to use distribution.log_prob
                xs = _torch.reshape(xs, (-1, 1))
            log_probs = distribution.log_prob(xs)
            pdfvals = _torch.exp(log_probs)
        elif pyrecest.backend.__backend_name__ == "jax":
            from jax.scipy.stats import (  # pylint: disable=import-error
                multivariate_normal,
            )

            if xs.ndim == 1 and self.dim == 1:
                # For 1-D distributions, we need to reshape the input to a 2-D tensor
                xs = reshape(xs, (-1, 1))

            pdfvals = multivariate_normal.pdf(xs, self.mu, self.C)
        else:
            raise NotImplementedError("Backend not supported")

        return pdfvals

    def shift(self, shift_by):
        """Return a copy with its mean shifted by ``shift_by``.

        Parameters
        ----------
        shift_by : array-like, shape (n,) or scalar
            Additive shift for the mean vector.
        """
        assert shift_by.ndim == 0 and self.dim == 1 or shift_by.shape[0] == self.dim
        new_gaussian = copy.deepcopy(self)
        new_gaussian.mu = self.mu + shift_by
        return new_gaussian

    def mean(self):
        """Return the mean vector with shape ``(n,)``."""
        return self.mu

    def mode(self, starting_point=None):
        """Return the mode of the Gaussian, equal to the mean vector."""
        _ = starting_point
        return self.mu

    def set_mode(self, new_mode):
        """Return a copy with a replaced mode.

        For a Gaussian distribution, the mode and mean are identical.
        """
        new_dist = copy.deepcopy(self)
        new_dist.mu = new_mode
        return new_dist

    def covariance(self):
        """Return the covariance matrix with shape ``(n, n)``."""
        return self.C

    def multiply(self, other):
        """Multiply two Gaussian densities and return the normalized product.

        Parameters
        ----------
        other : GaussianDistribution
            Gaussian distribution with the same dimension as ``self``.

        Returns
        -------
        GaussianDistribution
            Gaussian proportional to the pointwise product of both densities.
        """
        assert self.dim == other.dim
        self_precision = linalg.inv(self.C)
        other_precision = linalg.inv(other.C)
        new_C = linalg.inv(self_precision + other_precision)
        new_mu = matvec(
            new_C,
            matvec(self_precision, self.mu) + matvec(other_precision, other.mu),
        )
        return GaussianDistribution(new_mu, new_C, check_validity=False)

    def convolve(self, other):
        """Convolve two independent Gaussian distributions.

        Parameters
        ----------
        other : GaussianDistribution
            Gaussian distribution with the same dimension as ``self``.

        Returns
        -------
        GaussianDistribution
            Gaussian whose mean and covariance are the sums of both operands.
        """
        assert self.dim == other.dim
        new_mu = self.mu + other.mu
        new_C = self.C + other.C
        return GaussianDistribution(new_mu, new_C, check_validity=False)

    def marginalize_out(self, dimensions):
        """Return the marginal distribution after dropping dimensions.

        Parameters
        ----------
        dimensions : int or iterable of int
            Zero-based state dimensions to remove from the distribution.
        """
        if isinstance(dimensions, Integral):  # Make it iterable if single integer
            dimensions = [dimensions]
        else:
            dimensions = list(dimensions)

        assert all(
            isinstance(dim, Integral) and 0 <= dim < self.dim for dim in dimensions
        ), "Dimensions must be valid zero-based integer indices"

        dimensions = [int(dim) for dim in dimensions]
        remaining_dims = [i for i in range(self.dim) if i not in dimensions]
        new_mu = self.mu[remaining_dims]
        new_C = self.C[remaining_dims][
            :, remaining_dims
        ]  # Instead of np.ix_ for interface compatibiliy
        return GaussianDistribution(new_mu, new_C, check_validity=False)

    def sample(self, n):
        """Draw ``n`` random samples with shape ``(n, dim)``."""
        return random.multivariate_normal(mean=self.mu, cov=self.C, size=n)

    @staticmethod
    def from_distribution(distribution, check_validity=False):
        """Approximate or convert another distribution as a Gaussian.

        Gaussian mixtures are converted with ``to_gaussian``. Other
        distributions must expose mean and covariance information compatible
        with :class:`GaussianDistribution`.
        """
        from .gaussian_mixture import GaussianMixture

        if isinstance(distribution, GaussianMixture):
            gaussian = distribution.to_gaussian(check_validity=check_validity)
        else:
            mean = distribution.mean
            if callable(mean):
                mean = mean()

            covariance = distribution.covariance
            if callable(covariance):
                covariance = covariance()

            gaussian = GaussianDistribution(
                mean, covariance, check_validity=check_validity
            )
        return gaussian
