import unittest
from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import VonMisesFisherDistribution
from pyrecest.distributions import HypersphericalMixture
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import HyperhemisphericalGridDistribution
import numpy as np
import sys

class HyperhemisphericalGridDistributionTest(unittest.TestCase):
    @unittest.skipIf(sys.platform.startswith("win"), "requires Unix-based system")
    def test_approx_vmmixture_s2(self):
        dist1 = VonMisesFisherDistribution(1 / np.sqrt(2) * np.array([-1, 0, 1]), 2)
        dist2 = VonMisesFisherDistribution(1 / np.sqrt(2) * np.array([1, 0, -1]), 2)
        dist = HypersphericalMixture([dist1, dist2], [0.5, 0.5])

        hhgd = HyperhemisphericalGridDistribution.from_distribution(dist, 1012)
        np.testing.assert_almost_equal(hhgd.grid_values, 2 * dist.pdf(hhgd.get_grid()))
    """
    def test_multiply_vmf_mixture_s2(self):
        
        pass
    
    def test_multiply_vmf_mixture_s3(self):
        
        pass
    
    def test_multiply_error(self):
        dist1 = HypersphericalMixture(...)
        f1 = HyperhemisphericalGridDistribution.from_distribution(dist1, 84, 'eq_point_set')
        f2 = f1
        f2.grid_values = f2.grid_values[:-1]
        f2.grid = f2.grid[:, :-1]
        #with self.assertRaises(SomeException):
        f1.multiply(f2)

    def test_to_full_sphere(self):
        dist = HypersphericalMixture(...)
        hgd = HypersphericalGridDistribution.from_distribution(dist, 84, 'eq_point_set_symm')
        hhgd = HyperhemisphericalGridDistribution.from_distribution(dist, 42, 'eq_point_set_symm')

        hhgd2hgd = hhgd.to_full_sphere()
        self.assertEqual(hhgd2hgd.get_grid(), hgd.get_grid())
        self.assertEqual(hhgd2hgd.grid_values, hgd.grid_values)
    """

if __name__ == '__main__':
    unittest.main()
