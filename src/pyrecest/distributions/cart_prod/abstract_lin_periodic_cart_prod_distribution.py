from .abstract_lin_bounded_cart_prod_distribution import (
    AbstractLinBoundedCartProdDistribution,
)


class AbstractLinPeriodicCartProdDistribution(AbstractLinBoundedCartProdDistribution):
    """
    For Cartesian products of linear and periodic domains. Assumption is
    that it is bounded x R^n (in this order)
    """

    def __init__(self, bound_dim, lin_dim):
        AbstractLinBoundedCartProdDistribution.__init__(self, bound_dim, lin_dim)

    def get_manifold_size(self):
        if self.lin_dim <= 0:
            raise ValueError(
                "This class is not intended to be used for purely periodic domains."
            )
        return float("inf")
