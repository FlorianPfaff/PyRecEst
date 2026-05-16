import unittest

from pyrecest import backend
from pyrecest.backend import (
    all,
    allclose,
    cos,
    exp,
    linalg,
    mod,
    pi,
    random,
    squeeze,
    zeros,
)
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.cart_prod.gauss_von_mises_distribution import (
    GaussVonMisesDistribution,
)
from scipy.special import iv
from scipy.stats import multivariate_normal


class GaussVonMisesDistributionTest(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        self.g = GaussVonMisesDistribution(2, 1.3, 3, 0, 0.001, 0.7)
        self.testpoints = 2 * float(pi) * random.uniform(size=(2, 100))

    @staticmethod
    def _non_vectorized_pdf(gvm, xa):
        assert xa.shape[0] == gvm.mu.shape[0] + 1

        if xa.shape[1] > 1:
            p = zeros((1, xa.shape[1]))
            for i in range(xa.shape[1]):
                p[0, i] = GaussVonMisesDistributionTest._non_vectorized_pdf(
                    gvm, xa[:, [i]]
                )
            return p

        angle = xa[0, :]
        z = linalg.solve(gvm.A, xa[1:, :] - gvm.mu.reshape(-1, 1))
        Theta = gvm.alpha + gvm.beta @ z + 0.5 * z.T @ gvm.Gamma @ z
        p = (
            multivariate_normal.pdf(xa[1:, :].T, mean=gvm.mu.ravel(), cov=gvm.P)
            * exp(gvm.kappa * cos(angle - Theta))
            / (2.0 * float(pi) * iv(0, gvm.kappa))
        )
        return float(squeeze(p))

    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_pdf(self):
        self.assertTrue(
            allclose(
                self.g.pdf(self.testpoints),
                GaussVonMisesDistributionTest._non_vectorized_pdf(
                    self.g, self.testpoints
                ).ravel(),
                atol=1e-10,
            )
        )

    def test_integral(self):
        self.assertAlmostEqual(self.g.integrate(), 1, delta=1e-5)

    def test_mode(self):
        mode = self.g.mode()
        pdf_mode = self.g.pdf(mode)
        pdf_testpoints = self.g.pdf(self.testpoints)
        self.assertTrue(all(pdf_mode >= pdf_testpoints))

    def test_to_gaussian(self):
        gauss = self.g.to_gaussian()
        self.assertIsInstance(gauss, GaussianDistribution)
        self.assertTrue(allclose(gauss.mu, self.g.mode()))
        self.assertTrue(allclose(gauss.C[1:, 1:], self.g.P, atol=1e-10))

    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_sampling(self):
        # Deterministic Horwood sampler returns 2*lin_dim + 3 sigma points
        expected_n = 2 * self.g.lin_dim + 3
        d, w = self.g.sample_deterministic_horwood()
        self.assertEqual(d.shape, (self.g.lin_dim + self.g.bound_dim, expected_n))
        self.assertEqual(w.shape, (expected_n,))

        # Columns 1 and 2 are the +/-eta circular sigma points.  They must not
        # perturb the linear coordinate; the linear block remains at mu.
        self.assertTrue(allclose(d[1:, 1], self.g.mu, atol=1e-10))
        self.assertTrue(allclose(d[1:, 2], self.g.mu, atol=1e-10))

        # Columns 3.. contain linear sigma points.  Their circular coordinate
        # should equal get_theta evaluated at the transformed linear coordinate,
        # i.e. there must be no residual standardized linear-coordinate error.
        angle_error = mod(
            d[0, 3:] - self.g.get_theta(d[1:, 3:]) + float(pi),
            2.0 * float(pi),
        ) - float(pi)
        self.assertTrue(allclose(angle_error, zeros(expected_n - 3), atol=1e-10))

    def test_hybrid_moment(self):
        hm = self.g.hybrid_moment()
        self.assertEqual(hm.shape, (self.g.lin_dim + 2 * self.g.bound_dim,))
        hmn = self.g.hybrid_moment_numerical()
        self.assertEqual(hmn.shape, (self.g.lin_dim + 2 * self.g.bound_dim,))
        self.assertTrue(allclose(hm, hmn, atol=1e-5))
