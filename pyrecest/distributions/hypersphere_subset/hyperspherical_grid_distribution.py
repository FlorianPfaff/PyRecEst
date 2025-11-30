import numpy as np
import warnings
from .abstract_hypersphere_subset_grid_distribution import AbstractHypersphereSubsetGridDistribution
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
import matplotlib.pyplot as plt
from .hyperspherical_dirac_distribution import HypersphericalDiracDistribution
from ...sampling.hyperspherical_sampler import sample_sphere

class HypersphericalGridDistribution(AbstractHypersphereSubsetGridDistribution, AbstractHypersphericalDistribution):

    def mean_direction(self):
        mu = np.sum(self.grid * self.grid_values.T, axis=1)
        if np.linalg.norm(mu) < 1e-8:
            warnings.warn('Density may not actually have a mean direction because formula yields a point very close to the origin.')
        mu = mu / np.linalg.norm(mu)
        return mu

    def pdf(self, xs):
        assert self.grid_type == 'healpix', "Currently only supporting healpix grids."
        import healpy as hp
        indices = hp.vec2pix(self.grid_density_parameter["n_side"], xs[0,:], xs[1,:], xs[2,:])
        p = self.grid_values[indices]
        return p

    def plot(self):
        if self.dim == 3:
            AbstractHypersphericalDistribution.plot_sphere()
        plt.hold(True)
        hdd = HypersphericalDiracDistribution(self.grid, self.grid_values.T / np.sum(self.grid_values))
        h = hdd.plot()
        plt.hold(False)
        return h

    def plot_interpolated(self):
        from custom_hyperspherical_distribution import CustomHypersphericalDistribution
        chd = CustomHypersphericalDistribution(self.pdf, self.dim)
        h = chd.plot()
        return h

    def symmetrize(self):
        assert np.array_equal(self.grid[:, 1], -self.grid[:, int(self.grid.shape[1] / 2) + 1]), 'Symmetrize:AsymmetricGrid - Can only use symmetrize for symetric grids. Use eq_point_set_symm when calling fromDistribution or fromFunction.'
        grid_values_half = 0.5 * (self.grid_values[:int(self.grid.shape[1] / 2)] + self.grid_values[int(self.grid.shape[1] / 2):])
        hgd = HypersphericalGridDistribution(self.grid, np.concatenate([grid_values_half, grid_values_half]))
        return hgd

    def to_hemisphere(self, tol=1e-10):
        assert np.array_equal(self.grid[:, 1], -self.grid[:, int(self.grid.shape[1] / 2) + 1]), 'ToHemisphere:AsymmetricGrid - Can only use symmetrize for asymetric grids. Use eq_point_set_symm when calling fromDistribution or fromFunction.'
        if abs(self.grid_values[1] - self.grid_values[int(self.grid.shape[1] / 2) + 1]) < tol:
            grid_values_hemisphere = 2 * self.grid_values[:int(self.grid.shape[1] / 2)]
        else:
            warnings.warn('ToHemisphere:AsymmetricDensity - Density appears to be asymmetric. Not converting to hemispherical one.')
            grid_values_hemisphere = self.grid_values[:int(self.grid.shape[1] / 2)] + self.grid_values[int(self.grid.shape[1] / 2) + 1:]
        hgd = HypersphericalGridDistribution(self.grid, 2 * grid_values_hemisphere)
        return hgd
    
    @staticmethod
    def from_distribution(dist, grid_density_parameter, grid_type=None):  
        fun = dist.pdf
        sgd = HypersphericalGridDistribution.from_function(fun, grid_density_parameter, dist.dim, grid_type)
        return sgd

    @staticmethod
    def from_function(fun, grid_density_parameter, dim, grid_type=None):
        if grid_type is None:
            if dim == 2:
                grid_type = 'driscoll_healy'
            elif dim == 3:
                grid_type = 'healpix'
            else:
                raise NotImplementedError()
            
        grid, grid_specific_description = sample_sphere(grid_type, dim, grid_density_parameter)
        
        grid_values = fun(grid)
        hgd = HypersphericalGridDistribution(grid_values, grid_type=grid_type, grid=grid)
        hgd.grid_density_description.update(grid_specific_description)
        
        return hgd
        
    @staticmethod
    def get_leopardi_grid(dim, N):
    
        raise NotImplementedError()
        def eq_caps(dim, N):
            """
            Partition a sphere into nested spherical caps.
            """
            if dim == 1:
                # Circle, can be subdivided into equal areas easily
                sector = np.arange(1, N + 1)
                s_cap = sector * 2 * np.pi / N
                n_regions = np.ones_like(sector)
            elif N == 1:
                # Only single point. Can be anywhere.
                s_cap = np.array([np.pi])
                n_regions = np.array([1])
            else:
                c_polar = polar_colat(dim, N)
                ideal_angle = ideal_collar_angle(dim, N)
                n_collars = num_collars(N, c_polar, ideal_angle)
                r_regions = ideal_region_list(dim, N, c_polar, n_collars)

        
