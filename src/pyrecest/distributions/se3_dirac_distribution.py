# pylint: disable=no-name-in-module,no-member
from numbers import Integral

from pyrecest.backend import ones

from .abstract_se3_distribution import AbstractSE3Distribution
from .cart_prod.lin_hypersphere_cart_prod_dirac_distribution import (
    LinHypersphereCartProdDiracDistribution,
)
from .hypersphere_subset.hyperhemispherical_dirac_distribution import (
    HyperhemisphericalDiracDistribution,
)


class SE3DiracDistribution(
    LinHypersphereCartProdDiracDistribution, AbstractSE3Distribution
):
    def __init__(self, d, w=None):
        AbstractSE3Distribution.__init__(self)
        LinHypersphereCartProdDiracDistribution.__init__(self, bound_dim=3, d=d, w=w)

    def marginalize_linear(self):
        dist = HyperhemisphericalDiracDistribution(self.d[:, :4], self.w)
        return dist

    def mean(self):
        """
        Convenient access to hybrid_mean() to have a consistent interface
        throughout manifolds.

        :return: The mean of the distribution.
        :rtype:
        """
        m = self.hybrid_mean()
        return m

    @classmethod
    def from_distribution(cls, distribution, n_particles):
        if not isinstance(distribution, AbstractSE3Distribution):
            raise TypeError(
                "distribution must be an instance of AbstractSE3Distribution"
            )

        n_particles = cls._validate_particle_count(n_particles)

        ddist = cls(
            distribution.sample(n_particles),
            1 / n_particles * ones(n_particles),
        )
        return ddist

    @staticmethod
    def _validate_particle_count(n_particles):
        if (
            isinstance(n_particles, bool)
            or not isinstance(n_particles, Integral)
            or int(n_particles) <= 0
        ):
            raise ValueError("n_particles must be a positive integer")
        return int(n_particles)
