import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, pi
from pyrecest.distributions.cart_prod.mardia_sutton_distribution import (
    MardiaSuttonDistribution,
)
from pyrecest.distributions.circle.von_mises_distribution import VonMisesDistribution


class TestMardiaSuttonDistribution(unittest.TestCase):
    def setUp(self):
        self.mu = 2.0
        self.mu0 = 1.0
        self.kappa = 0.7
        self.rho1 = 0.5
        self.rho2 = 0.3
        self.sigma = 1.5
        self.dist = MardiaSuttonDistribution(
            self.mu, self.mu0, self.kappa, self.rho1, self.rho2, self.sigma
        )

    def test_instance(self):
        self.assertIsInstance(self.dist, MardiaSuttonDistribution)

    def test_parameters(self):
        npt.assert_allclose(self.dist.mu, self.mu)
        npt.assert_allclose(self.dist.mu0, self.mu0)
        npt.assert_allclose(self.dist.kappa, self.kappa)
        npt.assert_allclose(self.dist.rho1, self.rho1)
        npt.assert_allclose(self.dist.rho2, self.rho2)
        npt.assert_allclose(self.dist.sigma, self.sigma)

    def test_mu0_wrapping(self):
        dist2 = MardiaSuttonDistribution(
            self.mu,
            self.mu0 + 2.0 * float(pi),
            self.kappa,
            self.rho1,
            self.rho2,
            self.sigma,
        )
        npt.assert_allclose(dist2.mu0, self.dist.mu0, atol=1e-10)

    def test_pdf_positive(self):
        xs = array([[0.0, 0.0], [1.0, 2.0], [3.0, -1.0]])
        p = self.dist.pdf(xs)
        self.assertTrue((p > 0).all())

    def test_pdf_single_point(self):
        x = array([[1.0, 2.0]])
        p = self.dist.pdf(x)
        self.assertEqual(p.shape, (1,))
        self.assertTrue(float(p[0]) > 0)

    def test_pdf_normalization(self):
        import math

        from scipy.special import iv  # pylint: disable=no-name-in-module
        from scipy.stats import norm

        # At (mu0, mu), vm_part = exp(kappa) / (2*pi*I0(kappa))
        # and gaussian_part = 1 / (sqrt(2*pi) * sigmac)
        # muc = mu (since cos(mu0)-cos(mu0)=0 and sin(mu0)-sin(mu0)=0)
        rho = math.sqrt(self.rho1**2 + self.rho2**2)
        sigmac = self.sigma * math.sqrt(1.0 - rho**2)
        expected_vm = math.exp(self.kappa) / (2.0 * math.pi * iv(0, float(self.kappa)))
        expected_gauss = norm.pdf(self.mu, loc=self.mu, scale=sigmac)
        expected = expected_vm * expected_gauss

        p = self.dist.pdf(array([[self.mu0, self.mu]]))
        npt.assert_allclose(float(p[0]), expected, rtol=1e-6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_integral(self):
        self.assertAlmostEqual(self.dist.integrate(), 1.0, delta=1e-4)

    def test_mode(self):
        m = self.dist.mode()
        npt.assert_allclose(m, array([self.mu0, self.mu]))

    def test_linear_covariance(self):
        import math

        from scipy.special import iv  # pylint: disable=no-name-in-module

        bessel_ratio_1 = iv(1, self.kappa) / iv(0, self.kappa)
        bessel_ratio_2 = iv(2, self.kappa) / iv(0, self.kappa)

        rho_squared = self.rho1**2 + self.rho2**2
        conditional_variance = self.sigma**2 * (1.0 - rho_squared)
        aligned_rho_cos = self.rho1 * math.cos(self.mu0) + self.rho2 * math.sin(
            self.mu0
        )
        aligned_rho_sin = -self.rho1 * math.sin(self.mu0) + self.rho2 * math.cos(
            self.mu0
        )
        cos_variance = 0.5 * (1.0 + bessel_ratio_2) - bessel_ratio_1**2
        sin_variance = 0.5 * (1.0 - bessel_ratio_2)
        conditional_mean_variance = (
            self.sigma**2
            * self.kappa
            * (aligned_rho_cos**2 * cos_variance + aligned_rho_sin**2 * sin_variance)
        )
        expected = conditional_variance + conditional_mean_variance

        C = self.dist.linear_covariance()
        npt.assert_allclose(C, array([[expected]]))

    def test_marginalize_linear(self):
        vm = self.dist.marginalize_linear()
        self.assertIsInstance(vm, VonMisesDistribution)
        npt.assert_allclose(vm.mu, self.mu0)
        npt.assert_allclose(vm.kappa, self.kappa)

    def test_sample_shape(self):
        n = 100
        s = self.dist.sample(n)
        self.assertEqual(s.shape, (n, 2))

    def test_sample_circular_range(self):
        n = 500
        s = self.dist.sample(n)
        self.assertTrue((s[:, 0] >= 0).all())
        self.assertTrue((s[:, 0] < 2.0 * float(pi)).all())

    def test_invalid_kappa(self):
        with self.assertRaises(AssertionError):
            MardiaSuttonDistribution(
                self.mu, self.mu0, 0.0, self.rho1, self.rho2, self.sigma
            )

    def test_invalid_rho(self):
        with self.assertRaises(AssertionError):
            MardiaSuttonDistribution(
                self.mu, self.mu0, self.kappa, 0.8, 0.8, self.sigma
            )


if __name__ == "__main__":
    unittest.main()
