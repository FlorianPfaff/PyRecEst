import copy

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all as backend_all
from pyrecest.backend import array, isfinite, ones, reshape, stack, sum

from .abstract_linear_distribution import AbstractLinearDistribution
from .gaussian_distribution import GaussianDistribution, _validate_real_values
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
        _validate_real_values(new_mean, "new_mean")
        if not bool(backend_all(isfinite(new_mean))):
            raise ValueError("new_mean must contain only finite values.")

        new_mixture = copy.deepcopy(self)
        mean_offset = new_mean - self.mean()
        for dist in new_mixture.dists:
            dist.mu = dist.mu + mean_offset  # type: ignore
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
        means = array(means)
        if means.ndim == 0:
            means = reshape(means, (1, 1))
        elif means.ndim == 1:
            means = reshape(means, (-1, 1))
        elif means.ndim != 2:
            raise ValueError(
                "means must have shape (n_components, dim) or be a scalar/1D "
                "sequence of one-dimensional component means"
            )

        n_components, dim = means.shape
        covariance_matrices = array(covariance_matrices)
        expected_shape = (dim, dim, n_components)

        if covariance_matrices.ndim == 3:
            if covariance_matrices.shape != expected_shape:
                raise ValueError(
                    "covariance_matrices must have shape "
                    f"{expected_shape}, got {covariance_matrices.shape}"
                )
        elif n_components == 1 and covariance_matrices.shape == (dim, dim):
            covariance_matrices = reshape(covariance_matrices, expected_shape)
        elif dim == 1 and covariance_matrices.shape == (n_components,):
            covariance_matrices = reshape(covariance_matrices, expected_shape)
        elif dim == 1 and n_components == 1 and covariance_matrices.ndim == 0:
            covariance_matrices = reshape(covariance_matrices, expected_shape)
        else:
            raise ValueError(
                "covariance_matrices must have shape "
                f"{expected_shape}; a single ({dim}, {dim}) matrix is only "
                "accepted for one component"
            )

        if weights is None:
            weights = ones(n_components) / n_components
        else:
            weights = array(weights)
            if weights.ndim == 0:
                weights = reshape(weights, (1,))
            elif weights.ndim != 1:
                raise ValueError("weights must be scalar or one-dimensional")

        weights = LinearDiracDistribution._normalized_weights(weights)
        mu, C_from_means = LinearDiracDistribution.weighted_samples_to_mean_and_cov(
            means, weights
        )
        C_from_cov = sum(covariance_matrices * reshape(weights, (1, 1, -1)), axis=2)
        C = C_from_cov + C_from_means

        return mu, C
