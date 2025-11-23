import numpy as np
from ..abstract_periodic_grid_distribution import AbstractPeriodicGridDistribution
from .abstract_hypersphere_subset_distribution import AbstractHypersphereSubsetDistribution
from ..abstract_grid_distribution import AbstractGridDistribution

class AbstractHypersphereSubsetGridDistribution(AbstractPeriodicGridDistribution, AbstractHypersphereSubsetDistribution):
    def __init__(self, grid_values, grid_type = "custom", grid = None, enforce_pdf_nonnegative=True, dim=None):
        # Constructor
        if dim is None and grid is None or grid.ndim<=1:
            dim = 1
        elif dim is None:
            dim = grid.shape[1]

        AbstractPeriodicGridDistribution.__init__(self, dim)
        AbstractGridDistribution.__init__(self, grid_values, grid_type, grid, dim)
        self.enforce_pdf_nonnegative = enforce_pdf_nonnegative
        # Check if normalized. If not: Normalize
        self.normalize()

    def moment(self):
        C = self.grid @ np.diag(self.grid_values / np.sum(self.grid_values)) @ self.grid.T
        return C

    def normalize(self, tol=1e-2, warn_unnorm=True):
        f = AbstractGridDistribution.normalize(self, tol=tol, warn_unnorm=warn_unnorm)
        return f

