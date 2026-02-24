import numpy as np
import warnings
from .abstract_conditional_distribution import AbstractConditionalDistribution

class TdCondTdGridDistribution(AbstractConditionalDistribution):

    def multiply(self, other):
        assert np.all(self.grid == other.grid), "Multiply:IncompatibleGrid: Can only multiply for equal grids."
        warnings.warn("Multiply:UnnormalizedResult: Multiplication does not yield normalized result.")
        sdg = self
        sdg.fvals = sdg.fvals * other.fvals
        return sdg

    def marginalize_out(self, first_or_second):
        assert first_or_second in [1, 2], "firstOrSecond must be 1 or 2."
        raise NotImplementedError("Method is not implemented yet.")

    def fix_dim(self, first_or_second, point):
        assert first_or_second in [1, 2], "firstOrSecond must be 1 or 2."
        raise NotImplementedError("Method is not implemented yet.")

    def plot(self):
        if self.dim > 6:
            raise ValueError("Can currently only plot for T1, T2, and T3 torus.")
        raise NotImplementedError("Method is not implemented yet.")

    def plot_interpolated(self):
        if self.dim > 6:
            raise ValueError("Can currently only plot for T1, T2, and T3 torus.")
        raise NotImplementedError("Method is not implemented yet.")

    def get_manifold_size(self):
        raise ValueError("Not defined for conditional distributions because interpretation may not be 100% obvious.")

    @classmethod
    def from_function(cls, fun, no_of_grid_points, fun_does_cartesian_product, grid_type, dim):
        raise NotImplementedError("Method is not implemented yet.")