from ..hypersphere_subset.abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from ..nonperiodic.abstract_linear_distribution import AbstractLinearDistribution
from .cart_prod_stacked_distribution import CartProdStackedDistribution


class SE3LinVelCartProdStackedDistribution(CartProdStackedDistribution):
    def __init__(self, dists):
        if len(dists) != 2:
            raise ValueError("There must be exactly 2 distributions in dists.")
        if not isinstance(dists[0], AbstractHyperhemisphericalDistribution):
            raise TypeError(
                "The first distribution must be an instance of "
                "AbstractHyperhemisphericalDistribution."
            )
        if dists[0].input_dim != 4:
            raise ValueError("The first distribution must have input dimension 4.")
        if not isinstance(dists[1], AbstractLinearDistribution):
            raise TypeError(
                "The second distribution must be an instance of "
                "AbstractLinearDistribution."
            )
        if dists[1].dim != 6:
            raise ValueError("The second distribution must have 6 dimensions.")

        super().__init__(dists)
        self.boundD = dists[0].input_dim
        self.linD = dists[1].dim
        self.bound_dim = dists[0].dim
        self.lin_dim = dists[1].dim
        self.periodicManifoldType = "hyperhemisphere"

    @property
    def input_dim(self):
        return sum(dist.input_dim for dist in self.dists)

    def marginalize_linear(self):
        """Marginalize out linear velocity, retaining the orientation component."""
        return self.dists[0]

    def marginalize_periodic(self):
        """Marginalize out orientation, retaining the linear velocity component."""
        return self.dists[1]

    def get_manifold_size(self):
        return float("inf")
