import copy

# pylint: disable=no-name-in-module
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import dot, linalg, ndim, random, reshape
from scipy.linalg import cholesky

from .abstract_linear_distribution import AbstractLinearDistribution


class GaussianDistribution(AbstractLinearDistribution):
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
        new_dist = copy.deepcopy(self)
        new_dist.mu = new_mean
        return new_dist

    def pdf(self, xs):
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
        assert shift_by.ndim == 0 and self.dim == 1 or shift_by.shape[0] == self.dim
        new_gaussian = copy.deepcopy(self)
        new_gaussian.mu = self.mu + shift_by
        return new_gaussian

    def mean(self):
        return self.mu

    def mode(self, starting_point=None):
        _ = starting_point
        return self.mu

    def set_mode(self, new_mode):
        new_dist = copy.deepcopy(self)
        new_dist.mu = new_mode
        return new_dist

    def covariance(self):
        return self.C

    def multiply(self, other):
        assert self.dim == other.dim
        K = linalg.solve(self.C + other.C, self.C)
        new_mu = self.mu + dot(K, (other.mu - self.mu))
        new_C = self.C - dot(K, self.C)
        return GaussianDistribution(new_mu, new_C, check_validity=False)

    def convolve(self, other):
        assert self.dim == other.dim
        new_mu = self.mu + other.mu
        new_C = self.C + other.C
        return GaussianDistribution(new_mu, new_C, check_validity=False)

    def marginalize_out(self, dimensions):
        if isinstance(dimensions, int):  # Make it iterable if single integer
            dimensions = [dimensions]
        assert all(dim <= self.dim for dim in dimensions)
        remaining_dims = [i for i in range(self.dim) if i not in dimensions]
        new_mu = self.mu[remaining_dims]
        new_C = self.C[remaining_dims][
            :, remaining_dims
        ]  # Instead of np.ix_ for interface compatibiliy
        return GaussianDistribution(new_mu, new_C, check_validity=False)

    def sample(self, n):
        return random.multivariate_normal(mean=self.mu, cov=self.C, size=n)

    @staticmethod
    def from_distribution(distribution):
        from .gaussian_mixture import GaussianMixture

        if isinstance(distribution, GaussianMixture):
            gaussian = (
                distribution.to_gaussian()
            )  # Assuming to_gaussian method is defined in GaussianMixtureDistribution
        else:
            gaussian = GaussianDistribution(distribution.mean, distribution.covariance)
        return gaussian
