import numpy as np
import warnings
from .abstract_grid_distribution import AbstractGridDistribution
from .abstract_periodic_distribution import AbstractPeriodicDistribution
import copy


class AbstractBoundedGridDistribution(AbstractGridDistribution, AbstractPeriodicDistribution):
    def integrate(self, left=None, right=None):
        assert (left is None) == (
            right is None), 'Either both limits or only none should be None.'
        # Currently, all grids have aras of equal size
        if left is None and right is None:
            return self.get_manifold_size() * np.mean(self.grid_values)
        else:
            raise NotImplementedError("Integral for parts of the space are not yet implemented.")

    def normalize(self, tol=1e-4, warn_unnorm=True):
        integral = self.integrate()
        if np.any(self.grid_values < 0):
            warnings.warn('Normalization:negative',
                 'There are negative values. This usually points to a user error.')
        elif np.abs(integral) < 1e-200:
            raise ValueError('Normalization:almostZero',
                             'Sum of grid values is too close to zero, this usually points to a user error.')
        elif np.abs(integral - 1) > tol:
            if warn_unnorm:
                warnings.warn('Normalization:notNormalized',
                     'Grid values apparently do not belong to normalized density. Normalizing...')
        else:  # If < tolerance, just return a copy
            return copy.deepcopy(self)

        gd = copy.deepcopy(self)
        gd.grid_values = gd.grid_values / integral
        return gd

