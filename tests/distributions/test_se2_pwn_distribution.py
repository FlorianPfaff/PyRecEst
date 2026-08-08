import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.cart_prod.se2_pwn_distribution import SE2PWNDistribution


class TestSE2PWNDistribution(unittest.TestCase):
    def setUp(self):
        self.mu = array([1.0, 2.0, 3.0])
        self.C = array([[0.9, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
        self.dist = SE2PWNDistribution(self.mu, self.C)

    def test_constructor_stores_params(self):
        npt.assert_allclose(self.dist.mu, self.mu)
        npt.assert_allclose(self.dist.C, self.C)
        self.assertEqual(self.dist.bound_dim, 1)
        self.assertEqual(self.dist.lin_dim, 2)

    def test_pdf_positive(self):
        val = self.dist.pdf(self.mu.reshape(1, -1))
        self.assertGreater(float(val[0]), 0.0)

    def test_pdf_normalized(self):
        """Numerically verify that the pdf integrates to ~1 over its domain."""
        import scipy.integrate

        def integrand(x3, x2, x1):
            return float(self.dist.pdf(array([[x1, x2, x3]]))[0])

        mu = np.asarray(self.mu)
        C = np.asarray(self.C)
        spread = 5.0
        result, _ = scipy.integrate.tplquad(
            integrand,
            0.0,
            2.0 * np.pi,
            mu[1] - spread * np.sqrt(C[1, 1]),
            mu[1] + spread * np.sqrt(C[1, 1]),
            mu[2] - spread * np.sqrt(C[2, 2]),
            mu[2] + spread * np.sqrt(C[2, 2]),
            epsabs=1e-3,
            epsrel=1e-3,
        )
        npt.assert_allclose(result, 1.0, atol=1e-2)

    def test_mean4D_shape(self):
        m = self.dist.mean4D()
        self.assertEqual(m.shape, (4,))

    def test_mean4D_values(self):
        mu0 = float(self.mu[0])
        c00 = float(self.C[0, 0])
        expected = np.array(
            [
                np.cos(mu0) * np.exp(-c00 / 2),
                np.sin(mu0) * np.exp(-c00 / 2),
                float(self.mu[1]),
                float(self.mu[2]),
            ]
        )
        npt.assert_allclose(np.asarray(self.dist.mean4D()), expected, rtol=1e-6)

    def test_covariance4D_shape(self):
        cov = self.dist.covariance4D()
        self.assertEqual(cov.shape, (4, 4))

    def test_covariance4D_symmetric(self):
        cov = np.asarray(self.dist.covariance4D())
        npt.assert_allclose(cov, cov.T, atol=1e-12)

    def test_covariance4D_matches_numerical(self):
        """Analytical covariance should match a numerical estimate within tolerance."""
        np.random.seed(0)
        cov_analytical = np.asarray(self.dist.covariance4D())
        cov_numerical = np.asarray(self.dist.covariance4D_numerical(n_samples=100000))
        npt.assert_allclose(cov_analytical, cov_numerical, atol=5e-2)

    def test_from_samples_recovers_params(self):
        """from_samples should recover mu and C (up to Monte-Carlo noise)."""
        np.random.seed(42)
        samples = np.asarray(self.dist.sample(50000))
        fitted = SE2PWNDistribution.from_samples(samples)
        npt.assert_allclose(np.asarray(fitted.mu), np.asarray(self.mu), atol=0.05)
        npt.assert_allclose(np.asarray(fitted.C), np.asarray(self.C), atol=0.1)

    def test_from_samples_matches_empirical_population_covariance(self):
        """Moment matching must use empirical expectations normalized by n."""
        rng = np.random.default_rng(1234)
        source_mu = np.array([0.8, 1.2, -0.7])
        source_covariance = np.array(
            [
                [0.2, 0.05, -0.03],
                [0.05, 0.6, 0.1],
                [-0.03, 0.1, 0.4],
            ]
        )
        samples = rng.multivariate_normal(source_mu, source_covariance, size=64)
        samples[:, 0] %= 2.0 * np.pi

        fitted = SE2PWNDistribution.from_samples(samples)
        expected_linear_covariance = np.cov(samples[:, 1:].T, bias=True)

        npt.assert_allclose(
            np.asarray(fitted.C)[1:, 1:],
            expected_linear_covariance,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_sample_shape(self):
        s = self.dist.sample(10)
        self.assertEqual(s.shape, (10, 3))

    def test_sample_angle_in_range(self):
        s = np.asarray(self.dist.sample(500))
        self.assertTrue(np.all(s[:, 0] >= 0.0))
        self.assertTrue(np.all(s[:, 0] < 2.0 * np.pi))


if __name__ == "__main__":
    unittest.main()
