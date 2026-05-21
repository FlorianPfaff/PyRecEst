# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import linspace, pi
from pyrecest.distributions import CircularUniformDistribution

from .abstract_sampler import AbstractSampler


class AbstractHypertoroidalSampler(AbstractSampler):
    pass


class AbstractCircularSampler(AbstractHypertoroidalSampler):
    pass


class CircularUniformSampler(AbstractCircularSampler):
    def sample_stochastic(self, n_samples: int, dim: int = 1):
        assert (
            dim == 1
        ), "CircularUniformSampler is supposed to be used for the circle (which is one-dimensional) only."
        return CircularUniformDistribution().sample(n_samples)

    def get_grid(self, grid_density_parameter: int):
        """
        Returns an equidistant grid of points on the circle [0,2*pi).
        """
        grid_density_parameter = int(grid_density_parameter)
        if grid_density_parameter <= 0:
            raise ValueError("grid_density_parameter must be positive")
        points = linspace(0.0, 2.0 * pi, grid_density_parameter, endpoint=False)
        # Set it to the middle of the interval instead of the start
        points += (2.0 * pi / grid_density_parameter) / 2.0
        return points
