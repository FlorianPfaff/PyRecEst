import copy
import numpy as np
from .abstract_conditional_distribution import AbstractConditionalDistribution
from .abstract_grid_distribution import AbstractGridDistribution
from .hypersphere_subset.abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .hyperspherical_grid_distribution import HypersphericalGridDistribution

class SdCondSdGridDistribution(AbstractConditionalDistribution, AbstractGridDistribution):
    def __init__(self, grid_, gridValues_, enforcePdfNonnegative_=True):
        self.dim = 2 * grid_.shape[0]
        assert gridValues_.shape[0] == gridValues_.shape[1]
        assert grid_.shape[1] == gridValues_.shape[0]
        self.grid = grid_
        self.grid_values = gridValues_
        self.enforcePdfNonnegative = enforcePdfNonnegative_
        self.normalize()

    def normalize(self):
        tol = 0.01
        ints = np.mean(self.grid_values, axis=1) * AbstractHypersphericalDistribution.compute_unit_sphere_surface(self.dim // 2)
        if any(np.abs(ints - 1) > tol):
            if all(np.abs(ints - 1) <= tol):
                raise ValueError("Normalization:maybeWrongOrder: Not normalized but would be normalized if order of the spheres were swapped. Check input.")
            else:
                print("Normalization:notNormalized: When conditioning values for first sphere on second, normalization is not ensured. Check input or increase tolerance. No normalization is done, you may want to do this manually.")
        
    def multiply(self, other):
        assert np.all(self.grid == other.grid), "Multiply:IncompatibleGrid: Can only multiply for equal grids."
        print("Multiply:UnnormalizedResult: Multiplication does not yield normalized result.")
        sdg = copy.deepcopy(self)
        sdg.grid_values = sdg.grid_values * other.grid_values
        return sdg

    def marginalizeOut(self, firstOrSecond):
        if firstOrSecond == 1:
            gridValuesSgd = np.sum(self.gridValues, axis=1)
        elif firstOrSecond == 2:
            gridValuesSgd = np.sum(self.gridValues, axis=0)
        return HypersphericalGridDistribution(self.grid, gridValuesSgd)

    def fixDim(self, firstOrSecond, point):
        assert point.shape[0] == self.dim // 2
        lia, locb = np.isin(point.T, self.grid.T, assume_unique=True)
        if not lia:
            raise ValueError("Cannot fix value at this point because it is not on the grid")
        if firstOrSecond == 1:
            gridValuesSlice = self.gridValues[locb, :].T
        elif firstOrSecond == 2:
            gridValuesSlice = self.gridValues[:, locb]
        return HypersphericalGridDistribution(self.grid, gridValuesSlice)

    def plot(self):
        if self.dim != 4:
            raise ValueError("Can currently only plot for S2 sphere.")

        raise NotImplementedError("Method is not implemented yet.")

    def plotInterpolated(self):
        if self.dim != 4:
            raise ValueError("Can currently only plot for S2 sphere.")
        
        raise NotImplementedError("Method is not implemented yet.")

    def getManifoldSize(self):
        raise ValueError("Not defined for conditional distributions because there is some room for interpretation.")

    @staticmethod
    def fromFunction(fun, noOfGridPoints, funDoesCartesianProduct=False, gridType='eq_point_set', dim=6):
        raise NotImplementedError("Method is not implemented yet.")
