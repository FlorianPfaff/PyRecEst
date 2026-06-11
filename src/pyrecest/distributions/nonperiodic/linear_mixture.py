import warnings
from typing import Sequence

from ..abstract_mixture import AbstractMixture
from .abstract_linear_distribution import AbstractLinearDistribution
from .gaussian_distribution import GaussianDistribution


class LinearMixture(AbstractMixture, AbstractLinearDistribution):
    def __init__(self, dists: Sequence[AbstractLinearDistribution], w):
        from .gaussian_mixture import GaussianMixture

        if len(dists) == 0:
            raise ValueError("Mixture must contain at least one distribution")
        if not all(isinstance(dist, AbstractLinearDistribution) for dist in dists):
            raise ValueError("dists must be a list of linear distributions")
        if not isinstance(self, GaussianMixture) and all(
            isinstance(dist, GaussianDistribution) for dist in dists
        ):
            warnings.warn(
                "For mixtures of Gaussians, consider using GaussianMixture.",
                UserWarning,
            )
        AbstractLinearDistribution.__init__(self, dists[0].dim)
        AbstractMixture.__init__(self, dists, w)

    @property
    def input_dim(self):
        return AbstractLinearDistribution.input_dim.fget(self)
