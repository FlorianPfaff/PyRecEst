from pyrecest.backend import asarray, prod, stack

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
        return self.dists[1]

    def marginalize_periodic(self):
        return self.dists[0]

    def pdf(self, xs):
        xs = asarray(xs)
        ps = []
        next_dim = 0
        for dist in self.dists:
            next_input_dim = next_dim + dist.input_dim
            if xs.ndim == 1:
                xs_curr = xs[next_dim:next_input_dim]
            else:
                xs_curr = xs[:, next_dim:next_input_dim]
            ps.append(dist.pdf(xs_curr))
            next_dim = next_input_dim
        return prod(stack(ps), axis=0)

    def get_manifold_size(self):
        return float("inf")
