import warnings
from collections.abc import Callable

from .abstract_hemispherical_distribution import AbstractHemisphericalDistribution
from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .bingham_distribution import BinghamDistribution
from .custom_hyperhemispherical_distribution import CustomHyperhemisphericalDistribution


class CustomHemisphericalDistribution(
    CustomHyperhemisphericalDistribution, AbstractHemisphericalDistribution
):
    def __init__(self, f: Callable):
        AbstractHemisphericalDistribution.__init__(self)
        CustomHyperhemisphericalDistribution.__init__(self, f, 2)

    @classmethod
    def from_distribution(cls, distribution: "AbstractHypersphericalDistribution"):
        if distribution.dim != 2:
            raise ValueError("Dimension of the distribution should be 2.")

        if isinstance(distribution, AbstractHyperhemisphericalDistribution):
            return cls(distribution.pdf)
        if isinstance(distribution, BinghamDistribution):
            chsd = cls(distribution.pdf)
            chsd.scale_by = 2
            return chsd
        if isinstance(distribution, AbstractHypersphericalDistribution):
            warning_message = (
                "You are creating a CustomHyperhemispherical distribution based on a distribution on the full hypersphere. "
                + "Using numerical integration to calculate the normalization constant."
            )
            warnings.warn(warning_message, category=UserWarning)
            chhd_unnorm = CustomHyperhemisphericalDistribution(
                distribution.pdf, distribution.dim
            )
            norm_const_inv = chhd_unnorm.integrate()
            chsd = cls(distribution.pdf)
            chsd.scale_by = 1 / norm_const_inv
            return chsd

        raise ValueError("Input variable dist is of wrong class.")
