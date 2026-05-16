import copy
from collections.abc import Callable
from typing import Union

import matplotlib.pyplot as plt

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    arctan2,
    array,
    atleast_1d,
    exp,
    imag,
    int32,
    int64,
    mod,
    ones,
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

    @staticmethod
    def from_distribution(
        distribution: AbstractHypertoroidalDistribution, n_particles: int | None = None
    ):
        """Create a hypertoroidal Dirac approximation from a distribution.

        Grid distributions are converted deterministically by using every grid
        point as a Dirac location and normalized grid values as weights. Other
        hypertoroidal distributions are approximated by sampling and need
        ``n_particles``.
        """
        if not isinstance(distribution, AbstractHypertoroidalDistribution):
            raise ValueError(
                "from_distribution: invalidObject: First argument has to be "
                "a hypertoroidal distribution."
            )

        get_grid = getattr(distribution, "get_grid", None)
        if hasattr(distribution, "grid_values") and callable(get_grid):
            weights = reshape(distribution.grid_values, (-1,))
            weights = weights / sum(weights)
            return HypertoroidalDiracDistribution(
                get_grid(), weights, dim=distribution.dim
            )

        if n_particles is None:
            raise ValueError("n_particles is required for sampling-based conversion.")
        if n_particles <= 0:
            raise ValueError("n_particles must be a positive integer.")
        return HypertoroidalDiracDistribution(
            distribution.sample(n_particles),
            ones(n_particles) / n_particles,
            dim=distribution.dim,
        )

    def plot(self, resolution=128, **kwargs):
        _ = resolution
        assert self.dim <= 3, "Plotting not supported for this dimension"
        LinearDiracDistribution.plot(self, **kwargs)
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

    def apply_function(self, f: Callable, function_is_vectorized: bool = True):
        dist = super().apply_function(f, function_is_vectorized)
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

        try:
            dimensions = list(dimensions)
        except TypeError:
            dimensions = [dimensions]

        dimensions = [int(dim) for dim in dimensions]
        dimensions_to_remove = set(dimensions)

        if len(dimensions) != len(dimensions_to_remove):
            raise ValueError("dimensions must not contain duplicates.")

        if any(dim < 0 or dim >= self.dim for dim in dimensions_to_remove):
            raise ValueError(
                f"All marginalized-out dimensions must be in [0, {self.dim - 1}]."
            )

        if len(dimensions_to_remove) == 0:
            return copy.deepcopy(self)

        remaining_dims = [
            dim for dim in range(self.dim) if dim not in dimensions_to_remove
        ]

        if len(remaining_dims) == 0:
            raise ValueError("Cannot marginalize out all dimensions.")

        marginalized_particles = self.d[:, array(remaining_dims)]

        if len(remaining_dims) == 1:
            return CircularDiracDistribution(marginalized_particles[:, 0], self.w)

        return HypertoroidalDiracDistribution(
            marginalized_particles,
            self.w,
            dim=len(remaining_dims),
        )

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
