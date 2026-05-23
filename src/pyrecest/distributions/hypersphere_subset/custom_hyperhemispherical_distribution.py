from collections.abc import Callable
from typing import Union

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import asarray, int32, int64

from ..abstract_custom_distribution import AbstractCustomDistribution
from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .bingham_distribution import BinghamDistribution


class CustomHyperhemisphericalDistribution(
    AbstractCustomDistribution, AbstractHyperhemisphericalDistribution
):
    def __init__(self, f: Callable, dim: Union[int, int32, int64], scale_by: float = 1):
        """
        Initialize a CustomHyperhemisphericalDistribution.

        :param f: function to be used for custom distribution.
        :param dim: dimension of the distribution.
        :param scale_by: scaling factor for the distribution, default is 1.
        """
        AbstractHyperhemisphericalDistribution.__init__(self, dim=dim)
        AbstractCustomDistribution.__init__(self, f=f, scale_by=scale_by)

    def pdf(self, xs):
        """
        Calculate the probability density function at given points.

        :param xs: points where the pdf will be calculated.
        :return: numpy.ndarray: pdf values at given points.
        """
        assert xs.shape[-1] == self.dim + 1
        p = asarray(self.scale_by * self.f(xs))
        assert p.ndim <= 1, "Output format of pdf is not as expected"
        return p

    def integrate(self, integration_boundaries=None):
        """
        Integrate the pdf over given boundaries.

        :param integration_boundaries: boundaries of the integration, if None, the integration will be performed over the whole domain.
        :return: float: result of the integration.
        """
        return AbstractHyperhemisphericalDistribution.integrate(
            self, integration_boundaries
        )

    @staticmethod
    def from_distribution(distribution: "AbstractHypersphericalDistribution"):
        """
        Create a CustomHyperhemisphericalDistribution from another distribution.

        :param distribution: the distribution from which the CustomHyperhemisphericalDistribution will be created.
        :return: CustomHyperhemisphericalDistribution: the resulting distribution.
        :raises ValueError: if the type of dist is not supported.
        """
        if isinstance(distribution, AbstractHyperhemisphericalDistribution):
            return CustomHyperhemisphericalDistribution(
                distribution.pdf, distribution.dim
            )

        if isinstance(distribution, BinghamDistribution):
            chhd = CustomHyperhemisphericalDistribution(
                distribution.pdf, distribution.dim
            )
            chhd.scale_by = 2
            return chhd

        if isinstance(distribution, AbstractHypersphericalDistribution):
            chhd = CustomHyperhemisphericalDistribution(
                distribution.pdf, distribution.dim
            )
            norm_const_inv = chhd.integrate()
            chhd.scale_by = 1 / norm_const_inv
            return chhd

        raise ValueError("Input variable distribution is of the wrong class.")
