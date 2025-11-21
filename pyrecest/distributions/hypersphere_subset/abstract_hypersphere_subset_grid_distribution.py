import numpy as np
import warnings

from ..abstract_grid_distribution import AbstractGridDistribution 
from .abstract_hypersphere_subset_distribution import AbstractHypersphereSubsetDistribution
from .abstract_hyperhemispherical_distribution import AbstractHyperhemisphericalDistribution
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .von_mises_fisher_distribution import VonMisesFisherDistribution
from .bingham_distribution import BinghamDistribution
from .hyperspherical_mixture import HypersphericalMixture
from .watson_distribution import WatsonDistribution


class AbstractHypersphereSubsetGridDistribution(AbstractGridDistribution, AbstractHypersphereSubsetDistribution):
    
    def __init__(self, grid_, grid_values_, enforce_pdf_nonnegative=True):
        # Matlab convention: grid_ is (dim, N), grid_values_ is (N,)
        
        # Check size consistency
        if grid_.shape[1] != grid_values_.shape[0]:
            raise ValueError("Grid columns must match number of grid values.")
            
        self.dim = grid_.shape[0]
        self.grid = grid_
        self.grid_values = grid_values_
        self.enforce_pdf_nonnegative = enforce_pdf_nonnegative
        
        self.grid_type = 'unknown'
        
        self.normalize() 

    def mean_direction(self):
        warnings.warn("For hyperhemispheres, this function yields the mode and not the mean.", UserWarning)
        # If we took the mean, it would be biased toward [0;...;0;1]
        # because the lower half is considered inexistant.
        index_max = np.argmax(self.grid_values)
        mu = self.grid[:, index_max]
        return mu

    def moment(self):
        weights = self.grid_values / np.sum(self.grid_values) # (N,)
        
        # Weighted grid: (dim, N) via broadcasting
        weighted_grid = self.grid * weights 
        
        # C = grid * (gridValues' / sum(gridValues)) * grid'
        # Equivalent to sum(x_i * x_i.T * w_i)
        # C = weighted_grid @ grid.T -> (dim, N) @ (N, dim) -> (dim, dim)
        C = weighted_grid @ self.grid.T
        return C
    
    def normalize(self, tol=1e-2, warn_unnorm=True):
        # Delegates normalization to AbstractGridDistribution
        # Assuming AbstractGridDistribution implements normalize logic via super() call
        return super().normalize(tol=tol, warn_unnorm=warn_unnorm)

    def integral(self):
        # Delegates integral calculation to AbstractGridDistribution
        return super().integral()

    def multiply(self, other):
        # Check for grid compatibility
        if not np.array_equal(self.grid, other.grid):
             # Corresponding to Matlab error: 'Multiply:IncompatibleGrid'
             raise ValueError("Can only multiply for equal grids. Grids are incompatible.")
        
        # Delegates multiplication logic to AbstractGridDistribution
        return super().multiply(other)

    # --- Static Methods (Translated from Matlab implementation, assuming necessary imports) ---

    @staticmethod
    def from_distribution(distribution, no_of_grid_points, grid_type='healpix'):
        # Import here to avoid circular imports
        from .hyperhemispherical_grid_distribution import HyperhemisphericalGridDistribution
        from .hyperspherical_grid_distribution import HypersphericalGridDistribution
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
            raise ValueError('Converting a HypersphericalGridDistribution to a HyperhemisphericalGridDistribution is not supported')
        elif isinstance(distribution, AbstractHypersphericalDistribution):
            warnings.warn('Approximating a hyperspherical distribution on a hemisphere. The density may not be symmetric. Double check if this is intentional.',
                          UserWarning)
            fun = lambda x: 2 * distribution.pdf(x)
        else:
            raise ValueError('Distribution currently not supported.')
            
        # The concrete class constructor is called here, which makes this method non-abstract
        sgd = HyperhemisphericalGridDistribution.from_function(fun, no_of_grid_points, distribution.dim, grid_type)
        return sgd

