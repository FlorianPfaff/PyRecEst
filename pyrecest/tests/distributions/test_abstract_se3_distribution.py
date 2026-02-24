import unittest
import numpy as np
from pyrecest.distributions.se3_cart_prod_stacked_distribution import SE3CartProdStackedDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_watson_distribution import HyperhemisphericalWatsonDistribution
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.abstract_se3_distribution import AbstractSE3Distribution

class AbstractSE3DistributionTest(unittest.TestCase):

    def test_plot_mode(self):
        # Suppress warnings if necessary
        for i in range(10):
            cpd = SE3CartProdStackedDistribution([
                HyperhemisphericalWatsonDistribution(np.array([1, 1, 1 + i, 1]) / np.sqrt(3 + (1 + i) ** 2), 1),
                GaussianDistribution(np.array([1 + i / 2, i, 0]), np.diag([3, 2, 3]))
            ])
            cpd.plot_mode()

    def test_plot_state(self):
        # Suppress warnings if necessary
        cpd = SE3CartProdStackedDistribution([
            HyperhemisphericalWatsonDistribution(np.array([1, 1, 1, 1]) / np.sqrt(4), 20),
            GaussianDistribution(np.array([1, 0, 0]), np.block([[np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])]]))
        ])
        cpd.plot_state()

        # Suppress warnings if necessary
        cpd = SE3CartProdStackedDistribution([
            HyperhemisphericalWatsonDistribution(np.array([1, 1, 1, 1]) / np.sqrt(4), 20),
            GaussianDistribution(np.array([10, 10, 10]), np.block([[np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])]]))
        ])
        cpd.plot_state(10, False)

    def test_plot_trajectory(self):
        offsets = np.arange(10)
        quats = np.atleast_2d(np.array([1, 1, 1, 1])).T + np.vstack((offsets, np.zeros((3, 10))))
        quats = quats / np.linalg.norm(quats, axis=0)

        AbstractSE3Distribution.plot_trajectory(quats, np.atleast_2d(np.array([1, 2, 3])).T + np.vstack((offsets, np.zeros((1, 10)), offsets)))
        AbstractSE3Distribution.plot_trajectory(quats, np.atleast_2d(np.array([1, 2, 3])).T + np.vstack((offsets, np.zeros((1, 10)), offsets)), True, 0.05)

if __name__ == '__main__':
    unittest.main()
