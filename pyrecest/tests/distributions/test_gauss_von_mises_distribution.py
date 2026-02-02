import unittest
from scipy.stats import multivariate_normal
from scipy.special import iv

from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.cart_prod.gauss_von_mises_distribution import GaussVonMisesDistribution
from pyrecest.backend import pi, random, zeros, linalg, exp, cos, all
import numpy.testing as npt

class GaussVonMisesDistributionTest(unittest.TestCase):

    def setUp(self):
        self.g = GaussVonMisesDistribution(2, 1.3, 3, 0, 0.001, 0.7)
        self.testpoints = 2 * pi * random.rand(2, 100)
        random.seed(0)
        
    @staticmethod
    def _non_vectorized_pdf(gvm, xa):
        assert xa.shape[0] == gvm.mu.shape[0] + 1

        if xa.shape[1] > 1:
            p = zeros((1, xa.shape[1]))
            for i in range(xa.shape[1]):
                p[0, i] = GaussVonMisesDistributionTest._non_vectorized_pdf(gvm, xa[:, [i]])
            return p

        angle = xa[0, :]
        z = linalg.solve(gvm.A, xa[1:, :] - gvm.mu)
        Theta = gvm.alpha + gvm.beta.T @ z + 0.5 * z.T @ gvm.Gamma @ z
        p = multivariate_normal.pdf(xa[1:, :].T, mean=gvm.mu.ravel(), cov=gvm.P) * exp(gvm.kappa * cos(angle - Theta)) / (2 * pi * iv(0, gvm.kappa))
        return p

    def test_pdf(self):
        npt.assert_allclose(self.g.pdf(self.testpoints),
                               GaussVonMisesDistributionTest._non_vectorized_pdf(self.g, self.testpoints), atol=1e-10)

    def test_integral(self):
        self.assertAlmostEqual(self.g.integrate(), 1, delta=1e-5)

    def test_mode(self):
        mode = self.g.mode()
        self.assertTrue(all(self.g.pdf(mode) >= self.g.pdf(self.testpoints)))

    def test_to_gaussian(self):
        gauss = self.g.to_gaussian()
        self.assertIsInstance(gauss, GaussianDistribution)
        npt.assert_allclose(gauss.mu, self.g.mode())
        npt.assert_allclose(gauss.C[1:, 1:], self.g.P, atol=1e-10)

    def test_sampling(self):
        n = 10
        s = self.g.sample_deterministic_horwood()
        self.assertEqual(s.shape, (self.g.lin_dim + self.g.bound_dim, n))

    def test_hybrid_moment(self):
        hm = self.g.hybrid_moment()
        self.assertEqual(hm.shape, (self.g.lin_dim + 2 * self.g.bound_dim,))
        hmn = self.g.hybrid_moment_numerical()
        self.assertEqual(hmn.shape, (self.g.lin_dim + 2 * self.g.bound_dim,))
        npt.assert_allclose(hm, hmn, atol=1e-5)