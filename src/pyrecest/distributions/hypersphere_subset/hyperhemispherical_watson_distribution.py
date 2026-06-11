import copy
from typing import Union

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, concatenate, int32, int64, where, zeros

from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .watson_distribution import WatsonDistribution


class HyperhemisphericalWatsonDistribution(AbstractHyperhemisphericalDistribution):
    def __init__(self, mu, kappa):
        mu = array(mu)
        if mu.ndim != 1:
            raise ValueError("mu must be a 1-D vector")
        if bool(mu[-1] < 0):
            raise ValueError("mu must lie on the upper hyperhemisphere")
        self.dist_full_sphere = WatsonDistribution(mu, kappa)
        AbstractHyperhemisphericalDistribution.__init__(
            self, dim=self.dist_full_sphere.dim
        )

    def pdf(self, xs):
        return 2.0 * self.dist_full_sphere.pdf(xs)

    def set_mode(self, mu) -> "HyperhemisphericalWatsonDistribution":
        mu = array(mu)
        if mu.shape != self.mu.shape:
            raise ValueError("mu must have the same shape as the current mode")
        if bool(mu[-1] < 0):
            raise ValueError("mu must lie on the upper hyperhemisphere")
        dist = copy.deepcopy(self)
        dist.mu = copy.deepcopy(mu)
        return dist

    def sample(self, n: Union[int, int32, int64]):
        s_full = self.dist_full_sphere.sample(n)
        invert_mask = s_full[:, -1] < 0
        s = where(invert_mask[:, None], -s_full, s_full)
        return s

    @property
    def mu(self):
        return self.dist_full_sphere.mu

    @mu.setter
    def mu(self, mu):
        self.dist_full_sphere.mu = mu

    @property
    def kappa(self) -> float:
        return self.dist_full_sphere.kappa

    @kappa.setter
    def kappa(self, kappa: float):
        self.dist_full_sphere.kappa = kappa

    def mode(self):
        return self.mu

    def shift(self, shift_by) -> "HyperhemisphericalWatsonDistribution":
        canonical_mu = concatenate((zeros(self.input_dim - 1), array([1.0])))
        if not bool(allclose(self.mu, canonical_mu)):
            raise ValueError(
                "There is no true shifting for the hyperhemisphere. This is a "
                "function for compatibility and only works when mu is [0,0,...,1]."
            )
        return self.set_mode(shift_by)
