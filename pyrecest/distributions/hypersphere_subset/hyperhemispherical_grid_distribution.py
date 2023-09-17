from .abstract_hypersphere_subset_grid_distribution import AbstractHypersphereSubsetGridDistribution
from .abstract_hyperhemispherical_distribution import AbstractHyperhemisphericalDistribution
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
import numpy as np
from .hyperspherical_grid_distribution import HypersphericalGridDistribution
from .custom_hyperspherical_distribution import CustomHypersphericalDistribution
from .custom_hyperhemispherical_distribution import CustomHyperhemisphericalDistribution
from .hyperspherical_dirac_distribution import HypersphericalDiracDistribution
import warnings
from .spherical_harmonics_distribution_complex import SphericalHarmonicsDistributionComplex

class HyperhemisphericalGridDistribution(AbstractHypersphereSubsetGridDistribution, AbstractHyperhemisphericalDistribution):
    
    def __init__(self, grid_, grid_values_, enforce_pdf_nonnegative=True):
        assert np.all(grid_[-1, :] >= 0), "Always using upper hemisphere (along last dimension)."
        super().__init__(grid_, grid_values_, enforce_pdf_nonnegative)
    
    def mean_direction(self):
        # If we took the mean, it would be biased toward [0;...;0;1]
        # because the lower half is considered inexistant.
        index_max = np.argmax(self.grid_values)
        mu = self.grid[:, index_max]
        return mu
    
    def to_full_sphere(self):
        grid_ = np.hstack((self.grid, -self.grid))
        grid_values_ = 0.5 * np.hstack((self.grid_values, self.grid_values))
        hgd = HypersphericalGridDistribution(grid_, grid_values_)
        return hgd
    
    def plot(self):
        hdd = HypersphericalDiracDistribution(self.grid, self.grid_values.T)
        h = hdd.plot()
        return h
    
    def plot_interpolated(self):
        hdgd = self.to_full_sphere()
        hhgd_interp = CustomHyperhemisphericalDistribution(lambda x: 2 * hdgd.pdf(x), 3)
        h = hhgd_interp.plot()
        return h
    
    def plot_full_sphere_interpolated(self):
        if self.dim != 3:
            raise ValueError("Can currently only plot for hemisphere of S2 sphere.")
        hgd = self.to_full_sphere()
        shd = SphericalHarmonicsDistributionComplex.from_grid(hgd.grid_values, hgd.grid, 'identity')
        chhd = CustomHypersphericalDistribution.from_distribution(shd)
        h = chhd.plot()
        return h
    
    def get_closest_point(self, xs):
        all_distances = np.min(
            np.vstack((
                np.linalg.norm(self.grid.reshape(self.dim, 1, -1), xs, axis=1),
                np.linalg.norm(-self.grid.reshape(self.dim, 1, -1), xs, axis=1))),
            axis=0)

        indices = np.argmin(all_distances)
        points = self.get_grid_point(indices)
        return points, indices

    @staticmethod
    def from_distribution(distribution, no_of_grid_points, grid_type='healpix'):
        from .von_mises_fisher_distribution import VonMisesFisherDistribution
        from .bingham_distribution import BinghamDistribution
        from .hyperspherical_mixture import HypersphericalMixture
        from .watson_distribution import WatsonDistribution
        if isinstance(distribution, AbstractHyperhemisphericalDistribution):
            fun = distribution.pdf
        # pylint: disable=too-many-boolean-expressions
        elif (isinstance(distribution, WatsonDistribution) or 
              (isinstance(distribution, VonMisesFisherDistribution) and distribution.mu[-1] == 0) or 
              isinstance(distribution, BinghamDistribution) or
              (isinstance(distribution, HypersphericalMixture) and
               len(distribution.dists) == 2 and all([w == 0.5 for w in distribution.w]) and
               np.array_equal(distribution.dists[1].mu, -distribution.dists[0].mu))):
            fun = lambda x: 2 * distribution.pdf(x)
        elif isinstance(distribution, HypersphericalGridDistribution):
            raise ValueError('Converting a HypersphericalGridDistribution to a HypersphericalGridDistribution is not supported')
        elif isinstance(distribution, AbstractHypersphericalDistribution):
            warnings.warn('Approximating a hyperspherical distribution on a hemisphere. The density may not be symmetric. Double check if this is intentional.',
                          UserWarning)
            fun = lambda x: 2 * distribution.pdf(x)
        else:
            raise ValueError('Distribution currently not supported.')
        
        sgd = HyperhemisphericalGridDistribution.from_function(fun, no_of_grid_points, distribution.dim, grid_type)
        return sgd

    @staticmethod
    def from_function(fun, grid_density_parameter, dim=3, grid_type='healpix'):
        assert dim==3, "No other dimension currently supported"
        if grid_type == 'healpix':
            import healpy as hp # Import here so that one can use framework without healpy
            m = np.arange(grid_density_parameter)
            all_angs = hp.pix2ang(nside=grid_density_parameter, ipix=m)
            grid = hp.ang2vec(all_angs[0], all_angs[1])
        else:
            raise ValueError('Grid scheme not recognized')
        
        grid_values = np.array(fun(grid.T))
        sgd = HyperhemisphericalGridDistribution(grid, grid_values)
        return sgd