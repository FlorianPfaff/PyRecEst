import numpy as np
from .hyperhemispherical_grid_distribution import HyperhemisphericalGridDistribution
from .abstract_hemispherical_distribution import AbstractHemisphericalDistribution
from .spherical_grid_distribution import SphericalGridDistribution
from .custom_hemispherical_distribution import CustomHemisphericalDistribution

class HemisphericalGridDistribution(HyperhemisphericalGridDistribution, AbstractHemisphericalDistribution):
    def to_full_sphere(self):
        grid_ = np.hstack((self.grid, -self.grid))
        grid_values_ = 0.5 * np.vstack((self.grid_values, self.grid_values))
        hgd = SphericalGridDistribution(grid_, grid_values_)
        return hgd

    def plot_interpolated(self, use_harmonics=True):
        hdgd = self.to_full_sphere()

        def pdf_function(x):
            return 2 * hdgd.pdf(x, use_harmonics)

        hhgd_interp = CustomHemisphericalDistribution(pdf_function)

        h = hhgd_interp.plot()
        return h

    @staticmethod
    def from_distribution(dist, no_of_grid_points, grid_type='eq_point_set_symmetric'):
        if not (isinstance(no_of_grid_points, int) and no_of_grid_points > 0):
            raise ValueError("no_of_grid_points must be a positive integer")

        hgd = HyperhemisphericalGridDistribution.from_distribution(dist, no_of_grid_points, grid_type)
        sgd = HemisphericalGridDistribution(hgd.grid, hgd.grid_values, hgd.enforce_pdf_nonnegative)
        sgd.grid_type = hgd.grid_type
        return sgd

    @staticmethod
    def from_function(fun, no_of_grid_points, grid_type='eq_point_set_symm'):
        if not (isinstance(no_of_grid_points, int) and no_of_grid_points > 0):
            raise ValueError("no_of_grid_points must be a positive integer")

        # Always use adjusted version of eq_point_set since only this is suited to the hemihypersphere
        hgd = HyperhemisphericalGridDistribution.from_function(fun, no_of_grid_points, 3, grid_type)
        sgd = HemisphericalGridDistribution(hgd.grid, hgd.grid_values, hgd.enforce_pdf_nonnegative)
        sgd.grid_type = hgd.grid_type
        return sgd
