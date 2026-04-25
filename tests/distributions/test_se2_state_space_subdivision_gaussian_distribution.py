import unittest

import matplotlib.pyplot as plt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye

from pyrecest.distributions import GaussianDistribution, VonMisesDistribution
from pyrecest.distributions.cart_prod.se2_state_space_subdivision_gaussian_distribution import (
    SE2StateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.circle.circular_grid_distribution import CircularGridDistribution


class TestSE2StateSpaceSubdivisionGaussianDistribution(unittest.TestCase):
    def test_constructor(self):
        n = 100
        vm = VonMisesDistribution(0, 1)
        fig = CircularGridDistribution.from_distribution(vm, n)
        gaussians = [
            GaussianDistribution(array([0.0, 0.0]), eye(2)) for _ in range(n)
        ]
        SE2StateSpaceSubdivisionGaussianDistribution(fig, gaussians)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_plotting(self):
        plt.figure().add_subplot(111, projection="3d")

        n = 100
        vm = VonMisesDistribution(0, 1)
        fig = CircularGridDistribution.from_distribution(vm, n)
        gaussians = [
            GaussianDistribution(array([0.0, 0.0]), eye(2)) for _ in range(n)
        ]
        apd = SE2StateSpaceSubdivisionGaussianDistribution(fig, gaussians)
        apd.plot_state()
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
