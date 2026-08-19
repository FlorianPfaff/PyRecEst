from pyrecest.backend import array, reshape

from .abstract_custom_lin_bounded_cart_prod_distribution import (
    AbstractCustomLinBoundedCartProdDistribution,
)
from .abstract_hypercylindrical_distribution import AbstractHypercylindricalDistribution


class CustomHypercylindricalDistribution(
    AbstractCustomLinBoundedCartProdDistribution, AbstractHypercylindricalDistribution
):
    def __init__(self, f, bound_dim, lin_dim):
        """
        Constructor, it is the user's responsibility to ensure that f is a valid
        hypertoroidal density and takes arguments of the same form as
        .pdf, i.e., it needs to be vectorized.

        Parameters:
        f (function handle)
            pdf of the distribution
        bound_dim (scalar)
            dimension of the hypertorus
        lin_dim (scalar)
            linear dimension
        """
        AbstractHypercylindricalDistribution.__init__(self, bound_dim, lin_dim)
        AbstractCustomLinBoundedCartProdDistribution.__init__(
            self, f, bound_dim, lin_dim
        )

    @classmethod
    def from_distribution(cls, distribution):
        """
        Create a CustomHypercylindricalDistribution from another AbstractHypercylindricalDistribution.

        Parameters:
        dist (AbstractHypercylindricalDistribution)
            The distribution to convert into a CustomHypercylindricalDistribution

        Returns:
        chhd (CustomHypercylindricalDistribution)
            The created CustomHypercylindricalDistribution
        """
        return cls(distribution.pdf, distribution.bound_dim, distribution.lin_dim)

    def integrate(self, integration_boundaries=None):
        # Call the integrate method from the superclass
        return AbstractHypercylindricalDistribution.integrate(
            self, integration_boundaries
        )

    def linear_covariance_numerical(self, approximate_mean=None):
        """Return numerical linear covariance with a stable matrix shape."""
        covariance = AbstractHypercylindricalDistribution.linear_covariance_numerical(
            self, approximate_mean
        )
        return reshape(array(covariance), (self.lin_dim, self.lin_dim))

    def condition_on_periodic(self, input_periodic, normalize=True):
        # Call the condition_on_periodic method from the superclass
        return AbstractHypercylindricalDistribution.condition_on_periodic(
            self, input_periodic, normalize
        )

    def condition_on_linear(self, input_lin, normalize=True):
        # Call the condition_on_linear method from the superclass
        return AbstractHypercylindricalDistribution.condition_on_linear(
            self, input_lin, normalize
        )

    # Needed because abstract method needs to be implemented
    def marginalize_linear(self):
        return AbstractCustomLinBoundedCartProdDistribution.marginalize_linear(self)

    # Needed because abstract method needs to be implemented
    def marginalize_periodic(self):
        return AbstractCustomLinBoundedCartProdDistribution.marginalize_periodic(self)
