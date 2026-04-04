import copy
from collections.abc import Callable
from typing import Union

import matplotlib.pyplot as plt

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    arctan2,
    atleast_1d,
    exp,
    imag,
    int32,
    int64,
    mod,
    pi,
    real,
    reshape,
    sum,
    tile,
)

from ..abstract_dirac_distribution import AbstractDiracDistribution
from ..nonperiodic.linear_dirac_distribution import LinearDiracDistribution
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class HypertoroidalDiracDistribution(
    AbstractDiracDistribution, AbstractHypertoroidalDistribution
):
    def __init__(self, d, w=None, dim: int | None = None):
        """Can set dim manually to tell apart number of samples vs dimension for 1-D arrays."""
        if dim is None:
            if d.ndim > 1:
                dim = d.shape[-1]
            else:
                raise ValueError("Cannot automatically determine dimension.")

        AbstractHypertoroidalDistribution.__init__(self, dim)
        AbstractDiracDistribution.__init__(self, atleast_1d(mod(d, 2.0 * pi)), w=w)

    def plot(self):
        assert self.dim <= 3, "Plotting not supported for this dimension"
        LinearDiracDistribution.plot(self)
        if self.dim >= 1:
            plt.xlim(0, 2 * pi)
        if self.dim >= 2:
            plt.ylim(0, 2 * pi)
        if self.dim >= 3:
            ax = plt.gca()
            ax.set_zlim(0, 2 * pi)
        if self.dim >= 4:
            raise ValueError("Plotting not supported for this dimension")

    def set_mean(self, mean):
        dist = copy.deepcopy(self)
        dist.d = mod(dist.d - dist.mean_direction() + mean, 2.0 * pi)
        return dist

    def mean_direction(self):
        """
        Calculate the mean direction of the HypertoroidalDiracDistribution.

        :param self: HypertoroidalDiracDistribution instance
        :return: Mean direction
        """
        a = self.trigonometric_moment(1)
        m = mod(arctan2(imag(a), real(a)), 2.0 * pi)
        return m

    def trigonometric_moment(self, n: Union[int, int32, int64]):
        """
        Calculate the trigonometric moment of the HypertoroidalDiracDistribution.

        :param self: HypertoroidalDiracDistribution instance
        :param n: Integer moment order
        :return: Trigonometric moment
        """
        return sum(exp(1j * n * self.d.T) * tile(self.w, (self.dim, 1)), axis=1)

    def apply_function(self, f: Callable, f_supports_multiple: bool = True):
        dist = super().apply_function(f, f_supports_multiple)
        dist.d = mod(dist.d, 2.0 * pi)
        return dist

    def to_toroidal_wd(self):
        from .toroidal_dirac_distribution import ToroidalDiracDistribution

        assert self.dim == 2, "The dimension must be 2"
        twd = ToroidalDiracDistribution(self.d, self.w)
        return twd

    def marginalize_to_1D(self, dimension: Union[int, int32, int64]):
        from ..circle.circular_dirac_distribution import CircularDiracDistribution

        return CircularDiracDistribution(self.d[:, dimension], self.w)

    def marginalize_out(self, dimensions: int | list[int]):
        from ..circle.circular_dirac_distribution import CircularDiracDistribution

        remaining_dims = list(range(self.dim))
        remaining_dims = [dim for dim in remaining_dims if dim != dimensions]
        return CircularDiracDistribution(self.d[:, remaining_dims].squeeze(), self.w)

    def shift(self, shift_by) -> "HypertoroidalDiracDistribution":
        assert shift_by.shape[-1] == self.dim
        hd = copy.copy(self)
        hd.d = mod(self.d + reshape(shift_by, (1, -1)), 2.0 * pi)
        return hd

    def entropy(self):
        # Implement the entropy calculation here.
        raise NotImplementedError("Entropy calculation is not implemented")

    def to_wd(self):
        assert self.dim == 1
        from ..circle.circular_dirac_distribution import CircularDiracDistribution

        return CircularDiracDistribution(self.d, self.w)
