import unittest

import numpy.testing as npt
from pyrecest.backend import arange, linspace, pi
from pyrecest.distributions import VonMisesDistribution, WrappedNormalDistribution
from pyrecest.distributions.circle.circular_grid_distribution import (
    CircularGridDistribution,
)


class CircularGridDistributionTest(unittest.TestCase):
    @staticmethod
    def _test_grid_conversion(dist, coeffs, enforceNonnegative, tolerance):
        figd = CircularGridDistribution.from_distribution(
            dist, coeffs, enforce_pdf_nonnegative=enforceNonnegative
        )
        # Test grid values
        xvals = linspace(0, 2 * pi, coeffs, endpoint=False)
        npt.assert_allclose(figd.pdf(xvals), dist.pdf(xvals), atol=tolerance, rtol=0)
        # Test approximation of pdf
        xvals = arange(-2 * pi, 3 * pi, 0.01)
        npt.assert_allclose(figd.pdf(xvals), dist.pdf(xvals), atol=tolerance, rtol=0)

    def test_VMToGridId(self):
        mu = 0.4
        for kappa in arange(0.1, 2.1, 0.1):
            dist = VonMisesDistribution(mu, kappa)
            self._test_grid_conversion(dist, 101, False, 1e-6)

    def test_VMToGridSqrt(self):
        mu = 0.5
        for kappa in arange(0.1, 2.1, 0.1):
            dist = VonMisesDistribution(mu, kappa)
            self._test_grid_conversion(dist, 101, True, 1e-6)

    def test_WNToGridId(self):
        mu = 0.8
        for sigma in arange(0.2, 2.1, 0.1):
            dist = WrappedNormalDistribution(mu, sigma)
            self._test_grid_conversion(dist, 101, False, 3e-6)

    def test_WNToGridSqrt(self):
        mu = 0.9
        for sigma in arange(0.2, 2.1, 0.1):
            dist = WrappedNormalDistribution(mu, sigma)
            self._test_grid_conversion(dist, 101, True, 3e-6)

    def test_even_grid_trigonometric_moment(self):
        dist = VonMisesDistribution(0.4, 1.3)
        figd = CircularGridDistribution.from_distribution(dist, 100)
        npt.assert_allclose(
            figd.trigonometric_moment(1), dist.trigonometric_moment(1), atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
