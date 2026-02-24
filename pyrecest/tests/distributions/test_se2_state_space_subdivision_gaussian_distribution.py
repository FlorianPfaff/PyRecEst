import unittest
import numpy as np
from pyrecest.distributions.se2_state_space_subdivision_gaussian_distribution import SE2StateSpaceSubdivisionGaussianDistribution
from pyrecest.distributions import VonMisesDistribution, GaussianDistribution
from pyrecest.distributions.circle.circular_grid_distribution import CircularGridDistribution

class TestSE2StateSpaceSubdivisionGaussianDistribution(unittest.TestCase):

    def test_constructor(self):
        n = 100
        vm = VonMisesDistribution(0, 1)
        fig = CircularGridDistribution.from_distribution(vm, n)
        gaussians = np.array([GaussianDistribution(np.array([0, 0]), np.eye(2)) for _ in range(n)])
        SE2StateSpaceSubdivisionGaussianDistribution(fig, gaussians)

    def test_plotting(self):
        n = 100
        vm = VonMisesDistribution(0, 1)
        fig = CircularGridDistribution.from_distribution(vm, n)
        gaussians = np.array([GaussianDistribution(np.array([0, 0]), np.eye(2)) for _ in range(n)])
        apd = SE2StateSpaceSubdivisionGaussianDistribution(fig, gaussians)
        apd.plot_state()  # No assertion is needed, as we only check if the plotting function runs without errors


if __name__ == "__main__":
    unittest.main()
