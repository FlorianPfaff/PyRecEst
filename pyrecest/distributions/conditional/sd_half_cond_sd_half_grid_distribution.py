import numpy as np
from .abstract_conditional_distribution import AbstractConditionalDistribution
from ..abstract_grid_distribution import AbstractGridDistribution
from ..hypersphere_subset.hyperhemispherical_grid_distribution import HyperhemisphericalGridDistribution

class SdHalfCondSdHalfGridDistribution(AbstractConditionalDistribution, AbstractGridDistribution):
    def __init__(self, grid_, grid_values_, enforce_pdf_nonnegative=True):
        assert np.all(grid_[-1] >= 0), "Always using upper hemisphere (along last dimension)."
        self.dim = 2 * grid_.shape[0]
        assert grid_values_.shape[0] == grid_values_.shape[1]
        assert grid_.shape[1] == grid_values_.shape[0]
        self.grid = grid_
        self.grid_values = grid_values_
        self.enforce_pdf_nonnegative = enforce_pdf_nonnegative
        self.normalize()

    def normalize(self, tol=0.01):
        ints = np.mean(self.grid_values, axis=1) * 0.5 * self.compute_unit_sphere_surface(self.dim // 2)
        if any(np.abs(ints - 1) > tol):
            if all(np.abs(ints - 1) <= tol):
                raise ValueError("Not normalized but would be normalized if order of the spheres were swapped. Check input.")
            else:
                print("When conditioning values for first sphere on second, normalization is not ensured. One reason may be that you are approximating a density on the entire sphere that is not symmetrical. You can try to increase tolerance.")

    def multiply(self, other):
        assert np.array_equal(self.grid, other.grid), "Can only multiply for equal grids."
        print("Multiplication does not yield normalized result.")
        self.grid_values = self.grid_values * other.grid_values

    def marginalize_out(self, first_or_second):
        if first_or_second == 1:
            grid_values_sgd = np.sum(self.grid_values, axis=1).T
        elif first_or_second == 2:
            grid_values_sgd = np.sum(self.grid_values, axis=0)
        else:
            raise ValueError("Invalid value for first_or_second. Must be 1 or 2.")
        return HyperhemisphericalGridDistribution(self.grid, grid_values_sgd)

    def fix_dim(self, first_or_second, point):
        assert point.shape[0] == self.dim // 2
        lia, locb = ismember(point.T, self.grid.T, "rows")
        if not lia:
            raise ValueError("Cannot fix value at this point because it is not on the grid")
        if first_or_second == 1:
            grid_values_slice = self.grid_values[locb, :].T
        elif first_or_second == 2:
            grid_values_slice = self.grid_values[:, locb]
        else:
            raise ValueError("Invalid value for first_or_second. Must be 1 or 2.")
        return HyperhemisphericalGridDistribution(self.grid, grid_values_slice)