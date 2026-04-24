from pyrecest.backend import prod, stack

from ..hypersphere_subset.abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from ..nonperiodic.abstract_linear_distribution import AbstractLinearDistribution
from .cart_prod_stacked_distribution import CartProdStackedDistribution


class SE3LinVelCartProdStackedDistribution(CartProdStackedDistribution):
    def __init__(self, dists):
        assert len(dists) == 2, "There must be exactly 2 distributions in dists"
        assert dists[0].input_dim == 4, "The first distribution must have input dimension 4"
        assert isinstance(
            dists[0], AbstractHyperhemisphericalDistribution
        ), "The first distribution must be an instance of AbstractHyperhemisphericalDistribution"
        assert dists[1].dim == 6, "The second distribution must have 6 dimensions"
        assert isinstance(
            dists[1], AbstractLinearDistribution
        ), "The second distribution must be an instance of AbstractLinearDistribution"

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
