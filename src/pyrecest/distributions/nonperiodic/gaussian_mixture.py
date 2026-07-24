import copy

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all as backend_all
from pyrecest.backend import array, isfinite, ones, reshape, stack, sum

from .abstract_linear_distribution import AbstractLinearDistribution
from .gaussian_distribution import GaussianDistribution
from .linear_dirac_distribution import LinearDiracDistribution
from .linear_mixture import LinearMixture


class GaussianMixture(LinearMixture, AbstractLinearDistribution):
    def __init__(self, dists: list[GaussianDistribution], w):
        if len(dists) == 0:
            raise ValueError("Mixture must contain at least one distribution")
        if not all(isinstance(dist, GaussianDistribution) for dist in dists):
            raise ValueError("dists must be a list of GaussianDistribution instances")
        LinearMixture.__init__(self, dists, w)

    def mean(self):
        gauss_array = self.dists
        means = stack([g.mu for g in gauss_array], axis=0)  # shape (n, dim)
        return sum(means * reshape(self.w, (-1, 1)), axis=0)

    def set_mean(self, new_mean):
        new_mean = array(new_mean)
        if new_mean.ndim == 0:
            if self.dim != 1:
                raise ValueError(f"new_mean must have shape ({self.dim},), got scalar.")
            new_mean = reshape(new_mean, (1,))
        elif new_mean.shape != (self.dim,):
            raise ValueError(
                f"new_mean must have shape ({self.dim},), got {new_mean.shape}."
            )
        if not bool(backend_all(isfinite(new_mean))):
            raise ValueError("new_mean must contain only finite values.")

        new_mixture = copy.deepcopy(self)
        mean_offset = new_mean - self.mean()
        for dist in new_mixture.dists:
            dist.mu += mean_offset  # type: ignore
        return new_mixture

    def to_gaussian(self, check_validity=True):
        gauss_array = self.dists
        mu, C = self.mixture_parameters_to_gaussian_parameters(
            stack([g.mu for g in gauss_array], axis=0),
            stack([g.C for g in gauss_array], axis=2),
            self.w,
        )
        return GaussianDistribution(mu, C, check_validity=check_validity)

    def covariance(self):
        gauss_array = self.dists
        _, C = self.mixture_parameters_to_gaussian_parameters(
            stack([g.mu for g in gauss_array], axis=0),
            stack([g.C for g in gauss_array], axis=2),
            self.w,
        )
        return C

    @staticmethod
    def mixture_parameters_to_gaussian_parameters(
        means, covariance_matrices, weights=None
    ):
        if weights is None:
            weights = ones(means.shape[0]) / means.shape[0]
        else:
            weights = array(weights)

        C_from_cov = sum(covariance_matrices * reshape(weights, (1, 1, -1)), axis=2)
        mu, C_from_means = LinearDiracDistribution.weighted_samples_to_mean_and_cov(
            means, weights
        )
        C = C_from_cov + C_from_means

        return mu, C
