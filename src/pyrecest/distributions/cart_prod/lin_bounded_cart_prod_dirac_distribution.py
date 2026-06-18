import warnings
from abc import abstractmethod

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import concatenate

from ..abstract_dirac_distribution import AbstractDiracDistribution
from ..nonperiodic.linear_dirac_distribution import LinearDiracDistribution
from .abstract_lin_bounded_cart_prod_distribution import (
    AbstractLinBoundedCartProdDistribution,
)


class LinBoundedCartProdDiracDistribution(
    AbstractDiracDistribution, AbstractLinBoundedCartProdDistribution
):
    def marginalize_periodic(self):
        return LinearDiracDistribution(
            self.d[:, self.bound_dim :], self.w  # noqa: E203
        )

    def linear_mean(self):
        return self.marginalize_periodic().mean()

    def linear_covariance(
        self, approximate_mean=False
    ):  # This is just to be consistent with the interface
        if approximate_mean:
            warnings.warn(
                "Formula for mean is trivial, so approximate_mean is ignored."
            )
        return self.marginalize_periodic().covariance()

    @abstractmethod
    def marginalize_linear(self):
        pass

    def hybrid_mean(self):
        periodic = self.marginalize_linear()
        linear = self.marginalize_periodic()

        return concatenate((periodic.mean_direction(), linear.mean()))

    @classmethod
    def from_distribution(cls, distribution, n_particles):
        """
        Needs to be overwritten to allow the specification of bound_dim for Cartesian
        products of bounded and Euclidean manifolds
        """
        if not cls.is_valid_for_conversion(distribution):
            raise TypeError(
                "distribution is not valid for conversion to this Dirac type"
            )
        samples = distribution.sample(n_particles)
        return cls(distribution.bound_dim, samples)
