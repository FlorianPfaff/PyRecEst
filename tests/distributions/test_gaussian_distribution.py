import unittest

import numpy.testing as npt
import pyrecest.backend
import scipy

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, diag, linalg, linspace, matvec, to_numpy
from pyrecest.distributions import GaussianDistribution
from scipy.stats import multivariate_normal


class GaussianDistributionTest(unittest.TestCase):
    def test_gaussian_distribution_1d(self):
        g = GaussianDistribution(array(1.0), array(2.0))
        npt.assert_allclose(
            g.pdf(array([1.0, 2.0, 3.0])),
            multivariate_normal.pdf(array([1.0, 2.0, 3.0]), 1.0, 2.0),
            atol=1e-6,
        )

    def test_gaussian_distribution_3d(self):
        mu = array([2.0, 3.0, 4.0])
        C = array([[1.1, 0.4, 0.0], [0.4, 0.9, 0.0], [0.0, 0.0, 0.1]])
        g = GaussianDistribution(mu, C)

        xs = array(
            [
                [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ]
        ).T
        npt.assert_allclose(
            g.pdf(xs), multivariate_normal.pdf(xs, mu, C), atol=1e-10, rtol=5e-7
        )

        n = 10
        s = g.sample(n)
        self.assertEqual(
            s.shape,
            (
                n,
                3,
            ),
        )

    def test_mode(self):
        mu = array([1.0, 2.0, 3.0])
        C = array([[1.1, 0.4, 0.0], [0.4, 0.9, 0.0], [0.0, 0.0, 1.0]])
        g = GaussianDistribution(mu, C)

        self.assertTrue(allclose(g.mode(), mu, atol=1e-6))

    def test_shift(self):
        mu = array([3.0, 2.0, 1.0])
        C = array([[1.1, -0.4, 0.00], [-0.4, 0.9, 0.0], [0.0, 0.0, 1.0]])
        g = GaussianDistribution(mu, C)

        shift_by = array([2.0, -2.0, 3.0])
        g_shifted = g.shift(shift_by)

        self.assertTrue(allclose(g_shifted.mode(), mu + shift_by, atol=1e-6))

    def test_multiply_matches_information_form(self):
        distributions = [
            GaussianDistribution(array([0.0, 1.0]), diag(array([4.0, 1.0]))),
            GaussianDistribution(array([1.0, -0.5]), diag(array([1.0, 2.25]))),
            GaussianDistribution(array([-0.75, 0.25]), diag(array([0.5, 0.75]))),
        ]

        product = distributions[0]
        for distribution in distributions[1:]:
            product = product.multiply(distribution)

        precision_sum = sum(
            linalg.inv(distribution.C) for distribution in distributions
        )
        weighted_mean_sum = sum(
            matvec(linalg.inv(distribution.C), distribution.mu)
            for distribution in distributions
        )
        expected_covariance = linalg.inv(precision_sum)
        expected_mean = matvec(expected_covariance, weighted_mean_sum)

        npt.assert_allclose(to_numpy(product.mu), to_numpy(expected_mean), atol=1e-10)
        npt.assert_allclose(
            to_numpy(product.C), to_numpy(expected_covariance), atol=1e-10
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_marginalization(self):
        mu = array([1, 2])
        C = array([[1.1, 0.4], [0.4, 0.9]])
        g = GaussianDistribution(mu, C)

        grid = linspace(-10, 10, 30)
        dist_marginalized = g.marginalize_out(1)

        def marginalized_1D_via_integrate(xs):
            def integrand(y, x):
                return g.pdf(array([x, y]))

            result = []
            for x_curr in xs:
                integral_value, _ = scipy.integrate.quad(
                    integrand, -float("inf"), float("inf"), args=x_curr
                )
                result.append(integral_value)
            return array(result)

        self.assertTrue(
            allclose(
                dist_marginalized.pdf(grid),
                marginalized_1D_via_integrate(grid),
                atol=1e-9,
            )
        )

    def test_marginalize_out_rejects_invalid_dimensions(self):
        mu = array([1.0, 2.0])
        C = array([[1.1, 0.4], [0.4, 0.9]])
        g = GaussianDistribution(mu, C)

        with self.assertRaises(AssertionError):
            g.marginalize_out(-1)

        with self.assertRaises(AssertionError):
            g.marginalize_out(2)

        with self.assertRaises(AssertionError):
            g.marginalize_out([0, 2])


if __name__ == "__main__":
    unittest.main()
