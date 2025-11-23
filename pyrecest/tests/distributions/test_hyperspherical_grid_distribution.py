import numpy as np
import unittest
from pyrecest.distributions import VonMisesFisherDistribution, HypersphericalMixture
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import HypersphericalGridDistribution
import sys

class TestHypersphericalGridDistribution(unittest.TestCase):

    def test_approx_vmmixture_s2(self):
        dist1 = VonMisesFisherDistribution(1 / np.sqrt(2) * np.array([-1, 0, 1]), 2)
        dist2 = VonMisesFisherDistribution(np.array([0, -1, 0]), 2)
        dist = HypersphericalMixture([dist1, dist2], [0.5, 0.5])

        hgd = HypersphericalGridDistribution.from_distribution(dist, 8)
        np.testing.assert_almost_equal(hgd.pdf(hgd.get_grid()), dist.pdf(hgd.get_grid()))
    
    def test_mean_direction_s2(self):
        mu = np.array([1,2,3])/np.linalg.norm([1,2,3])
        vmf = VonMisesFisherDistribution(mu, 2)
        hgd = HypersphericalGridDistribution.from_distribution(vmf, 5)
        np.testing.assert_almost_equal(hgd.mean_direction(), mu, decimal=2)
    
    def test_multiply_vmfs2(self):
        for kappa1 in np.arange(0.1, 4.1, 0.3):
            for kappa2 in np.arange(0.1, 4.1, 0.3):
                dist1 = VonMisesFisherDistribution(1 / np.sqrt(2) * np.array([-1, 0, 1]), kappa1)
                dist2 = VonMisesFisherDistribution(np.array([0, -1, 0]), kappa2)
                dist_result = dist1.multiply(dist2)
                
                hgd1 = HypersphericalGridDistribution.from_distribution(dist1, 15, 'healpix')
                hgd2 = HypersphericalGridDistribution.from_distribution(dist2, 15, 'healpix')
                hgd_filtered = hgd1.multiply(hgd2)
                
                np.testing.assert_almost_equal(hgd_filtered.pdf(hgd1.get_grid()), dist_result.pdf(hgd1.get_grid()), decimal=4)
    
    @unittest.skipIf(sys.platform.startswith("win"), "requires Unix-based system")
    def test_healpix_closest_point(self):
        import healpy as hp
        nside = 4
        n_grid_points = hp.nside2npix(nside)

        # Generate HEALPix grid in Cartesian coordinates
        x, y, z = hp.pix2vec(nside, np.arange(n_grid_points))
        grid = np.vstack((x,y,z))
        
        # Find the closest point for all grid points
        closest_pixel_indices = hp.vec2pix(nside, grid[0,:], grid[1,:], grid[2,:])

        # Verify that the result is 0...(n_grid_points-1)
        expected_pixel_indices = np.arange(n_grid_points)
        self.assertTrue(np.array_equal(closest_pixel_indices, expected_pixel_indices))


if __name__ == '__main__':
    unittest.main()
