from pyrecest.distributions.cart_prod.state_space_subdivision_distribution import StateSpaceSubdivisionDistribution
from pyrecest.distributions.cart_prod.abstract_hypercylindrical_distribution import AbstractHypercylindricalDistribution
from pyrecest.distributions.nonperiodic.custom_linear_distribution import CustomLinearDistribution
from pyrecest.distributions.circle.circular_uniform_distribution import CircularUniformDistribution
from scipy.integrate import quad
import numpy as np

class HypercylindricalStateSpaceSubdivisionDistribution(StateSpaceSubdivisionDistribution, AbstractHypercylindricalDistribution):

    def __init__(self, gd_, lin_distributions):
        StateSpaceSubdivisionDistribution.__init__(self, gd_, lin_distributions)

    def plot(self, interpolate=False):
        if interpolate:
            return AbstractHypercylindricalDistribution.plot(self)
        else:
            return StateSpaceSubdivisionDistribution.plot(self)

    def plot_interpolated(self):
        return self.plot(interpolate=True)

    def mode(self):
        return StateSpaceSubdivisionDistribution.mode(self)

    @staticmethod
    def from_distribution(distribution, no_of_grid_points, grid_type='CartesianProd'):
        return HypercylindricalStateSpaceSubdivisionDistribution.from_function(
            distribution.pdf, no_of_grid_points, distribution.linD, distribution.boundD, grid_type)

    @staticmethod
    def from_function(fun, no_of_grid_points, dim_lin, dim_bound=1, grid_type='CartesianProd', int_range=(-np.inf , np.inf)):
        assert dim_lin == 1, 'Currently, bounded dimension must be 1.'

        gd = CircularGridDistribution.from_distribution(CircularUniformDistribution(), no_of_grid_points)
        grid = gd.get_grid()
        cds = [None] * no_of_grid_points

        for i in range(no_of_grid_points):
            fun_curr = lambda y: np.reshape(fun(np.vstack((grid[i] * np.ones_like(y), y))), np.shape(y))

            # Obtain grid value via integral
            gd.grid_values[i], _ = quad(fun_curr, int_range[0], int_range[1])

            # Original function divided by grid value is linear
            cds[i] = CustomLinearDistribution(lambda x: fun_curr(x) / gd.grid_values[i], 1)

        return HypercylindricalStateSpaceSubdivisionDistribution(gd, cds)
