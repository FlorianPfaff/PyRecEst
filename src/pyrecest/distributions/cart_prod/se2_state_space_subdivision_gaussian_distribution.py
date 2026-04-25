from pyrecest.distributions.abstract_se2_distribution import AbstractSE2Distribution
from pyrecest.distributions.cart_prod.hypercylindrical_state_space_subdivision_gaussian_distribution import (
    HypercylindricalStateSpaceSubdivisionGaussianDistribution,
)


class SE2StateSpaceSubdivisionGaussianDistribution(
    HypercylindricalStateSpaceSubdivisionGaussianDistribution, AbstractSE2Distribution
):
    pass
