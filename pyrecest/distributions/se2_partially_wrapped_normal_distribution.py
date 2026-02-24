from .abstract_se2_distribution import AbstractSE2Distribution
from .cart_prod.partially_wrapped_normal_distribution import PartiallyWrappedNormalDistribution

class SE2PartiallyWrappedNormalDistribution(PartiallyWrappedNormalDistribution, AbstractSE2Distribution):

    def __init__(self, mu, C):
        AbstractSE2Distribution.__init__(self)
        PartiallyWrappedNormalDistribution.__init__(self, mu, C, bound_dim=self.bound_dim)