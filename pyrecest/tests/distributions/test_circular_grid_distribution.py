import unittest
import numpy as np
from pyrecest.distributions.circle.circular_grid_distribution import CircularGridDistribution
from pyrecest.distributions import VonMisesDistribution, WrappedNormalDistribution

class CircularGridDistributionTest(unittest.TestCase):

    @staticmethod
    def _test_grid_conversion(dist, coeffs, enforceNonnegative, tolerance):
        figd = CircularGridDistribution.from_distribution(dist, coeffs, enforce_pdf_nonnegative=enforceNonnegative)
        # Test grid values
        xvals = np.linspace(0, 2*np.pi, coeffs, endpoint=False)
        np.testing.assert_allclose(figd.pdf(xvals), dist.pdf(xvals), atol=tolerance)
        # Test approximation of pdf
        xvals = np.arange(-2 * np.pi, 3 * np.pi, 0.01)
        np.testing.assert_allclose(figd.pdf(xvals), dist.pdf(xvals), atol=tolerance)

    def test_VMToGridId(self):
        mu = 0.4
        for kappa in np.arange(.1, 2.1, .1):
            dist = VonMisesDistribution(mu, kappa)
            self._test_grid_conversion(dist, 101, False, 1E-8)

    def test_VMToGridSqrt(self):
        mu = 0.5
        for kappa in np.arange(.1, 2.1, .1):
            dist = VonMisesDistribution(mu, kappa)
            self._test_grid_conversion(dist, 101, True, 1E-8)

    def test_WNToGridId(self):
        mu = 0.8
        for sigma in np.arange(.2, 2.1, .1):
            dist = WrappedNormalDistribution(mu, sigma)
            self._test_grid_conversion(dist, 101, False, 1E-8)

    def test_WNToGridSqrt(self):
        mu = 0.9
        for sigma in np.arange(.2, 2.1, .1):
            dist = WrappedNormalDistribution(mu, sigma)
            self._test_grid_conversion(dist, 101, True, 1E-8)

if __name__ == "__main__":
    unittest.main()