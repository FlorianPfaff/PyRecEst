import unittest

import numpy.testing as npt

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import allclose, arange, array, exp, ones_like, pi, sqrt, sum
from pyrecest.distributions import WrappedNormalDistribution


class WrappedNormalDistributionTest(unittest.TestCase):
    def setUp(self):
        self.mu = array(3.0)
        self.sigma = array(1.5)
        self.wn = WrappedNormalDistribution(self.mu, self.sigma)

    def test_pdf_values_are_as_expected(self):
        """
        Test that the probability density function (pdf) returns expected values.
        """

        def approx_with_wrapping(x):
            k = arange(-20, 21)
            total = sum(exp(-((x - self.mu + 2 * pi * k) ** 2) / (2 * self.sigma**2)))
            return 1 / sqrt(2 * pi) / self.sigma * total

        test_points = [self.mu, self.mu - 1, self.mu + 2]
        for point in test_points:
            with self.subTest(x=point):
                npt.assert_almost_equal(
                    self.wn.pdf(point), approx_with_wrapping(point), decimal=7
                )

        x = arange(0, 7)
        self.assertTrue(
            allclose(
                self.wn.pdf(x),
                array([approx_with_wrapping(xi) for xi in x]),
                rtol=1e-7,
            )
        )

    def test_pdf_with_large_sigma_is_uniform(self):
        """
        Test that the pdf with large sigma is approximately a uniform distribution.
        """
        wn_large_sigma = WrappedNormalDistribution(array(0.0), array(100.0))
        x = arange(0, 7)
        fx = ones_like(x) / (2.0 * pi)
        self.assertTrue(allclose(wn_large_sigma.pdf(x), fx, rtol=1e-10))


if __name__ == "__main__":
    unittest.main()
