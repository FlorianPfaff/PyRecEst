import warnings
import numpy as np
from .abstract_filter_type import AbstractFilterType
from pyrecest.distributions.abstract_grid_distribution import AbstractGridDistribution

class AbstractGridFilter(AbstractFilterType):
    def __init__(self, gd):
        if not isinstance(gd, AbstractGridDistribution):
            raise ValueError("gd must be an instance of AbstractGridDistribution")
        self.filter_state = gd

    @property 
    def filter_state(self) -> AbstractGridDistribution:
        """
        Get the current estimate of the grid distribution.
        Returns : AbstractGridDistribution : the current grid distribution
        """
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if not isinstance(new_state, AbstractGridDistribution):
            warnings.warn("gd_ is not a GridDistribution. Transforming the distribution with a number of coefficients equal to that of the filter.", RuntimeWarning)
            new_state = self.filter_state.from_distribution(new_state, len(self.filter_state.grid_values), self.filter_state.enforce_pdf_nonnegative)

        elif np.shape(self.filter_state.grid_values) != np.shape(new_state.grid_values):
            warnings.warn("New grid has a different number of grid points.", RuntimeWarning)

        self._filter_state = new_state

    def update_nonlinear(self, likelihood, z):
        grid_vals_new = self.filter_state.grid_values * np.reshape(likelihood(z, self.filter_state.get_grid()), (-1, 1))
        assert np.shape(grid_vals_new) == np.shape(self.filter_state.grid_values)

        self.filter_state.grid_values = grid_vals_new
        self.filter_state = self.filter_state.normalize(suppress_warning = True)

    def plot_filter_state(self):
        self.filter_state.plot_state()