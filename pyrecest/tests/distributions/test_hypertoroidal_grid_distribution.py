import numpy as np
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import HypertoroidalGridDistribution
import unittest
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import ToroidalWrappedNormalDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_wrapped_normal_distribution import HypertoroidalWrappedNormalDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_mixture import HypertoroidalMixture


class HypertoroidalGridDistributionTest(unittest.TestCase):

    def test_get_grid(self):
        grid = np.array([[1, 2, 3, 4], [1, 2, 3, 4]]).T
        hgd = HypertoroidalGridDistribution(np.array([1, 1, 1, 1]) / ((2 * np.pi) ** 2), grid=grid)
        np.testing.assert_allclose(hgd.get_grid(), grid)
        np.testing.assert_allclose(np.shape(hgd.get_grid()), (4, 2))

    def test_approx_vmmixture_t2(self):
        dist = HypertoroidalMixture([
            ToroidalWrappedNormalDistribution(np.array([1,1]),np.array([[1,0.5],[0.5,1]])),
            ToroidalWrappedNormalDistribution(np.array([3,3]),np.array([[1,-0.5],[-0.5,1]]))], 
            np.array([0.5,0.5]))

        hgd = HypertoroidalGridDistribution.from_distribution(dist, 31)
        np.testing.assert_almost_equal(hgd.grid_values, dist.pdf(hgd.get_grid()), decimal=6)
        self.assertTrue((np.min(hgd.get_grid(), axis=0) == np.array([0, 0])).all())
        self.assertTrue((np.max(hgd.get_grid(), axis=0) > 6).all())
        self.assertEqual(hgd.grid_type, 'CartesianProd')

    def test_from_function_3D(self):
        np.random.seed(0)
        test_points = np.random.rand(3, 30)
        C = np.array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]])
        mu = 2 * np.pi * np.random.rand(3, 1)
        hwnd = HypertoroidalWrappedNormalDistribution(mu, C)
        n_grid_points = [27, 27, 27]
        hfd_id = HypertoroidalGridDistribution.from_function(hwnd.pdf, n_grid_points, 3, 'CartesianProd')
        self.assertIsInstance(hfd_id, HypertoroidalGridDistribution)
        np.testing.assert_almost_equal(hfd_id.pdf(test_points), hwnd.pdf(test_points), decimal=6)


if __name__ == '__main__':
    unittest.main()
