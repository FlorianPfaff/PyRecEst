from ..abstract_custom_distribution import AbstractCustomDistribution
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class CustomHypersphericalDistribution(
    AbstractCustomDistribution, AbstractHypersphericalDistribution
):
    def __init__(self, f, dim, scale_by=1):
        AbstractCustomDistribution.__init__(self, f, scale_by)
        AbstractHypersphericalDistribution.__init__(self, dim)

    @staticmethod
    def from_distribution(distribution):
        if not isinstance(distribution, AbstractHypersphericalDistribution):
            raise ValueError("Input variable distribution is of the wrong class.")

        chd = CustomHypersphericalDistribution(distribution.pdf, distribution.dim)
        return chd

    def integrate(self, integration_boundaries=None):
        return AbstractHypersphericalDistribution.integrate(
            self, integration_boundaries
        )
