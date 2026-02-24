import unittest
import numpy as np
from pyrecest.distributions.cart_prod.se3_lin_vel_cart_prod_stacked_distribution import SE3LinVelCartProdStackedDistribution
from pyrecest.distributions import HyperhemisphericalUniformDistribution, GaussianDistribution, HyperhemisphericalWatsonDistribution


class TestSE3LinVelCartProdStackedDistribution(unittest.TestCase):

    def test_constructor(self):
        SE3LinVelCartProdStackedDistribution([HyperhemisphericalUniformDistribution(4), GaussianDistribution(np.array([1, 2, 3, 4, 5, 6]), np.diag([3, 2, 1, 4, 3, 4]))])

    def test_sampling(self):
        cpd = SE3LinVelCartProdStackedDistribution([HyperhemisphericalUniformDistribution(4), GaussianDistribution(np.array([1, 2, 0, -2, -1, 3]), np.diag([3, 2, 3, 3, 4, 5]))])
        samples = cpd.sample(100)
        self.assertEqual(samples.shape, (10, 100))

    def test_pdf(self):
        cpd = SE3LinVelCartProdStackedDistribution([HyperhemisphericalUniformDistribution(4), GaussianDistribution(np.array([1, 2, 0, -2, -1, 3]), np.diag([3, 2, 3, 3, 4, 5]))])
        self.assertEqual(cpd.pdf(np.random.randn(10, 100)).shape, (1, 100))

        pdf_values = cpd.pdf(np.ones((10, 100)))
        self.assertTrue(np.allclose(np.diff(pdf_values), np.zeros(99)))

    def test_mode(self):
        watson = HyperhemisphericalWatsonDistribution(np.array([2, 1, 3, 1]) / np.linalg.norm(np.array([2, 1, 3, 1])), 2)
        gaussian = GaussianDistribution(np.array([1, 2, 0, -2, -1, 3]), np.diag([3, 2, 3, 3, 4, 5]))
        cpd = SE3LinVelCartProdStackedDistribution([watson, gaussian])
        self.assertTrue(np.allclose(cpd.mode(), np.hstack([watson.mode(), gaussian.mode()])))


if __name__ == "__main__":
    unittest.main()
