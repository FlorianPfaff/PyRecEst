import unittest
from pyrecest.distributions import (
                         HyperhemisphericalUniformDistribution, GaussianDistribution,
                         VonMisesFisherDistribution, HypersphericalMixture)
from pyrecest.distributions.cart_prod.state_space_subdivision_distribution import StateSpaceSubdivisionDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import HyperhemisphericalGridDistribution
from pyrecest.distributions.hypersphere_subset.spherical_grid_distribution import SphericalGridDistribution
import numpy as np

class TestStateSpaceSubdivisionDistribution(unittest.TestCase):
    def test_normalization(self):
        dist = HypersphericalMixture(
            [
                VonMisesFisherDistribution(1 / np.sqrt(2) * np.array([-1, 0, 1]), 2),
                VonMisesFisherDistribution(np.array([0, -1, 0]), 2)
            ],
            [0.5, 0.5]
        )
        sgd = SphericalGridDistribution.from_distribution(dist, 1012, "eq_point_set")
        self.assertAlmostEqual(sgd.integral, 1, delta=1e-2)
    
    def test_plot_h2x_r2(self):
        n = 100
        means = np.random.rand(2, n)
        vars = np.eye(2) + np.random.rand(1, 1, n)
        linear_distributions = [GaussianDistribution(means[:, i], vars[:, :, i]) for i in range(n)]
        gd = HyperhemisphericalGridDistribution.from_distribution(
            HyperhemisphericalUniformDistribution(3), n)
        rbd = StateSpaceSubdivisionDistribution(gd, linear_distributions)
        h = rbd.plot()

        with self.assertWarns(StateSpaceSubdivisionDistribution.invalidPlotArgument):  # Replace with the appropriate warning
            rbd.plot(True)