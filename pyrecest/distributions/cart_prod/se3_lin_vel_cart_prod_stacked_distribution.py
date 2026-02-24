from .cart_prod.cart_prod_stacked_distribution import CartProdStackedDistribution
from .abstract_se3_lin_vel_distribution import AbstractSE3LinVelDistribution
from .hypersphere_subset.abstract_hyperhemispherical_distribution import AbstractHyperhemisphericalDistribution
from .nonperiodic.abstract_linear_distribution import AbstractLinearDistribution
import numpy as np

class SE3LinVelCartProdStackedDistribution(CartProdStackedDistribution, AbstractSE3LinVelDistribution):
    def __init__(self, dists):
        assert len(dists) == 2, "There must be exactly 2 distributions in dists"
        assert dists[0].dim == 4, "The first distribution must have 4 dimensions"
        assert isinstance(dists[0], AbstractHyperhemisphericalDistribution), "The first distribution must be an instance of AbstractHyperhemisphericalDistribution"
        assert dists[1].dim == 6, "The second distribution must have 6 dimensions"
        assert isinstance(dists[1], AbstractLinearDistribution), "The second distribution must be an instance of AbstractLinearDistribution"

        super().__init__(dists)
        self.boundD = dists[0].dim
        self.linD = dists[1].dim
        self.periodicManifoldType = "hyperhemisphere"

    def marginalize_linear(self):
        return self.dists[0]

    def marginalize_periodic(self):
        return self.dists[1]
    
    def get_manifold_size(self):
        return np.inf

