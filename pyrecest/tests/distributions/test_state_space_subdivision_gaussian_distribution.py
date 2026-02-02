import unittest
import numpy as np
from pyrecest.distributions.circle.circular_uniform_distribution import CircularUniformDistribution
from pyrecest.distributions import GaussianDistribution, CircularFourierDistribution, VonMisesDistribution
from pyrecest.distributions.circle.circular_grid_distribution import CircularGridDistribution
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import StateSpaceSubdivisionGaussianDistribution

class StateSpaceSubdivisionGaussianDistributionTest(unittest.TestCase):

    def test_multiply_s1xr1_identical_precise(self):
        n = 100
        rbd1 = StateSpaceSubdivisionGaussianDistribution(CircularFourierDistribution.from_distribution(
            CircularUniformDistribution, n),
            np.tile(GaussianDistribution(0, 1), (n,)))
        
        rbd2 = StateSpaceSubdivisionGaussianDistribution(CircularFourierDistribution.from_distribution(
            CircularUniformDistribution, n),
            np.tile(GaussianDistribution(2, 1), (n,)))
        
        rbdUp = rbd1.multiply(rbd2)
        for i in range(100):
            self.assertLess(np.linalg.det(rbdUp.linear_distributions[0].C), np.linalg.det(rbd1.linear_distributions[0].C))
            self.assertEqual(rbdUp.linear_distributions[i].mean, 1)

    # TODO more tests

    def test_mode(self):
        n = 100
        mu_periodic = 4
        mu_linear = np.array([1, 2, 3])
        rbd = StateSpaceSubdivisionGaussianDistribution(CircularGridDistribution.from_distribution(
            VonMisesDistribution(mu_periodic, 10), n),
            np.tile(GaussianDistribution(mu_linear, np.eye(3)), (n,)))

        # Since we land somewhere on the grid, it is expected we that
        # we are not indefinitely precise.
        np.testing.assert_allclose(rbd.mode(), np.hstack((mu_periodic, mu_linear)), atol=np.pi / n)

if __name__ == '__main__':
    unittest.main()
