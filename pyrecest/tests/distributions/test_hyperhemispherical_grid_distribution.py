= test file =
import unittest
import numpy as np
import sys
import warnings

# Assume pyrecest classes are available
from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import VonMisesFisherDistribution
from pyrecest.distributions import HypersphericalMixture
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import HyperhemisphericalGridDistribution
from pyrecest.distributions.hyperspherical_grid_distribution import HypersphericalGridDistribution
from pyrecest.distributions.bingham_distribution import BinghamDistribution
# Assuming other necessary imports like WatsonDistribution, etc., exist if needed for other tests

# Using nside=16 for healpix density parameter (3072 total pixels on S2)
N_SIDE = 16 

class HyperhemisphericalGridDistributionTest(unittest.TestCase):
    
    @unittest.skipIf(sys.platform.startswith("win"), "requires Unix-based system for healpy/grid generation")
    def test_warning_asymm(self):
        # Corresponds to Matlab testWarningAsymm: Testing warning for asymmetric VMF/Mixture approximation
        vmf_asymmetric = VonMisesFisherDistribution(1 / np.sqrt(2) * np.array([-1, 0, -1]), 2)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # This VMF is non-antipodally symmetric, so approximating it on the hemisphere should warn.
            HyperhemisphericalGridDistribution.from_distribution(vmf_asymmetric, N_SIDE, 'healpix')
            
            # Check if warning was raised (checking for keywords from the Matlab warning message)
            warning_message = "Approximating a hyperspherical distribution on a hemisphere"
            self.assertTrue(any(warning_message in str(warn.message) for warn in w))

    @unittest.skipIf(sys.platform.startswith("win"), "requires Unix-based system for healpy/grid generation")
    def test_approx_vmmixture_s2(self):
        # Corresponds to Matlab testApproxVMMixtureS2
        # Use an antipodally symmetric mixture
        dist1 = VonMisesFisherDistribution(1 / np.sqrt(2) * np.array([-1, 0, 1]), 2)
        dist2 = VonMisesFisherDistribution(1 / np.sqrt(2) * np.array([1, 0, -1]), 2)
        dist = HypersphericalMixture([dist1, dist2], [0.5, 0.5])

        # Grid density parameter is N_SIDE
        hhgd = HyperhemisphericalGridDistribution.from_distribution(dist, N_SIDE, 'healpix')
        
        # Verify grid values are 2 * PDF(grid_points). Note PDF expects (N, dim), grid is (dim, N)
        np.testing.assert_almost_equal(hhgd.grid_values, 2 * dist.pdf(hhgd.get_grid().T))
        
    @unittest.skipIf(sys.platform.startswith("win"), "requires Unix-based system for healpy/grid generation")
    def test_approx_bingham_s2(self):
        # Corresponds to Matlab testApproxBinghamS2
        M = np.eye(3)
        Z = np.array([-2, -1, 0])
        dist = BinghamDistribution(Z, M)
        
        hhgd = HyperhemisphericalGridDistribution.from_distribution(dist, N_SIDE, 'healpix')
        
        # Bingham is antipodally symmetric, so grid values should be 2 * PDF(grid_points)
        np.testing.assert_almost_equal(hhgd.grid_values, 2 * dist.pdf(hhgd.get_grid().T))
        
    # Skipping testApproxBinghamS3 because the Python implementation only supports dim=3 for healpix grid generation.

    @unittest.skipIf(sys.platform.startswith("win"), "requires Unix-based system for healpy/grid generation")
    def test_multiply_vmf_mixture_s2(self):
        # Corresponds to Matlab testMultiplyVMFMixtureS2
        kappa1, kappa2 = 2.0, 1.0
        
        dist1 = HypersphericalMixture([
            VonMisesFisherDistribution(1 / np.sqrt(2) * np.array([-1, 0, 1]), kappa1),
            VonMisesFisherDistribution(-1 / np.sqrt(2) * np.array([-1, 0, 1]), kappa1)
        ], [0.5, 0.5])
        
        dist2 = HypersphericalMixture([
            VonMisesFisherDistribution(np.array([0, -1, 0]), kappa2),
            VonMisesFisherDistribution(np.array([0, 1, 0]), kappa2)
        ], [0.5, 0.5])
        
        hhgd1 = HyperhemisphericalGridDistribution.from_distribution(dist1, N_SIDE, 'healpix')
        hhgd2 = HyperhemisphericalGridDistribution.from_distribution(dist2, N_SIDE, 'healpix')
        hhgd_filtered = hhgd1.multiply(hhgd2)
        
        # Reference full sphere multiplication result
        hgd1 = HypersphericalGridDistribution.from_distribution(dist1, N_SIDE, 'healpix')
        hgd2 = HypersphericalGridDistribution.from_distribution(dist2, N_SIDE, 'healpix')
        hgd_filtered = hgd1.multiply(hgd2)
        
        hgd_grid = hgd_filtered.get_grid()
        
        # Identify upper hemisphere points in the full grid
        upper_hemisphere_indices = np.where(hgd_grid[-1, :] >= 0)[0]
        
        # Check shapes match
        self.assertEqual(hhgd_filtered.get_grid().shape[1], len(upper_hemisphere_indices))
        
        # Verify scaling: 0.5 * P_half = P_full (for corresponding upper hemisphere points)
        hgd_values_upper_half = hgd_filtered.grid_values[upper_hemisphere_indices]
        np.testing.assert_almost_equal(0.5 * hhgd_filtered.grid_values, hgd_values_upper_half, decimal=4)
        
    # Skipping testMultiplyVMFMixtureS3 due to dim=4 constraint

    def test_multiply_error(self):
        # Corresponds to Matlab testMultiplyError: Incompatible Grid size check
        dist1 = HypersphericalMixture([
            VonMisesFisherDistribution(1 / np.sqrt(2) * np.array([-1, 0, 1]), 1),
            VonMisesFisherDistribution(-1 / np.sqrt(2) * np.array([-1, 0, 1]), 1)
        ], [0.5, 0.5])
        
        f1 = HyperhemisphericalGridDistribution.from_distribution(dist1, N_SIDE, 'healpix')
        
        # Create an incompatible distribution f2 by manually trimming the grid/values
        N_points_f1 = f1.get_grid().shape[1]
        self.assertTrue(N_points_f1 > 1)

        grid_incompatible = f1.get_grid()[:, :-1]
        values_incompatible = f1.grid_values[:-1]
        
        # Create a new incompatible instance
        f2_incompatible = HyperhemisphericalGridDistribution(grid_incompatible, values_incompatible, True)
        
        # Test multiplication error
        with self.assertRaisesRegex(ValueError, "incompatible"):
             f1.multiply(f2_incompatible)

    @unittest.skipIf(sys.platform.startswith("win"), "requires Unix-based system for healpy/grid generation")
    def test_to_full_sphere(self):
        # Corresponds to Matlab testToFullSphere
        dist = HypersphericalMixture([
            VonMisesFisherDistribution(1 / np.sqrt(2) * np.array([-1, 0, 1]), 1),
            VonMisesFisherDistribution(-1 / np.sqrt(2) * np.array([-1, 0, 1]), 1)
        ], [0.5, 0.5])
        
        hgd = HypersphericalGridDistribution.from_distribution(dist, N_SIDE, 'healpix') 
        hhgd = HyperhemisphericalGridDistribution.from_distribution(dist, N_SIDE, 'healpix')
        
        hhgd2hgd = hhgd.to_full_sphere()
        
        # Find corresponding indices in hgd for upper hemisphere
        hgd_grid = hgd.get_grid()
        upper_hemisphere_indices = np.where(hgd_grid[-1, :] >= 0)[0]
        
        # Check that the number of points is doubled
        self.assertEqual(hhgd2hgd.get_grid().shape[1], 2 * hhgd.get_grid().shape[1])
        
        # Check values: hhgd2hgd values should match the density values of hgd (full sphere)
        hgd_values_upper_half = hgd.grid_values[upper_hemisphere_indices]
        
        # The first half of hhgd2hgd values should match hgd upper half values
        N_half = len(hhgd.grid_values)
        np.testing.assert_almost_equal(hhgd2hgd.grid_values[:N_half], hgd_values_upper_half, decimal=4)
        
        # The second half should also match due to symmetry assumption in toFullSphere conversion
        np.testing.assert_almost_equal(hhgd2hgd.grid_values[N_half:], hgd_values_upper_half, decimal=4)

if __name__ == '__main__':
    # Using specific argv to avoid interaction when running in non-standard environments
    unittest.main(argv=['first-arg-is-ignored'], exit=False)