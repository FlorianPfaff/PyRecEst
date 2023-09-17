from .hyperspherical_grid_distribution import HypersphericalGridDistribution
from .abstract_spherical_distribution import AbstractSphericalDistribution
import numpy as np


class SphericalGridDistribution(HypersphericalGridDistribution, AbstractSphericalDistribution):
    def __init__(self, grid, grid_values, enforce_pdf_nonnegative=True, grid_type='unknown'):
        if np.any(grid < -1) or np.any(1 < grid):
            raise ValueError("Grid values must be between -1 and 1")
        if np.any(grid_values < 0):
            raise ValueError("Grid values must be non-negative")
        
        AbstractSphericalDistribution.__init__(self)
        HypersphericalGridDistribution.__init__(self, grid_values, grid_type, grid, enforce_pdf_nonnegative, dim=self.dim)
        

    def normalize(self, tol=1e-2, warn_unnorm=True):
        raise NotImplementedError("Normalization not implemented for SphericalGridDistribution")

    def plot_interpolated(self, use_harmonics=True):
        raise NotImplementedError("Plotting not implemented for SphericalGridDistribution")

    def pdf(self, xa, use_harmonics=True):
        raise NotImplementedError("PDF not implemented for SphericalGridDistribution")