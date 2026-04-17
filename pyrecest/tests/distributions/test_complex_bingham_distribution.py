import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    diag,
    exp,
    linalg,
    log,
    mean,
    ones,
    pi,
    random,
    real,
    sort,
    sqrt,
)
from pyrecest.distributions import ComplexBinghamDistribution


class TestComplexBinghamDistribution(unittest.TestCase):
    """Tests for ComplexBinghamDistribution."""

    def setUp(self):
        # Simple 2x2 diagonal Hermitian B
        self.B2 = diag(array([-3.0, 0.0], dtype=complex))
        self.cB2 = ComplexBinghamDistribution(self.B2)

        # 3x3 diagonal Hermitian B
        self.B3 = diag(array([-5.0, -2.0, 0.0], dtype=complex))
        self.cB3 = ComplexBinghamDistribution(self.B3)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_constructor_hermitian_check(self):
        """Non-Hermitian matrix should raise AssertionError."""
        with self.assertRaises(AssertionError):
            ComplexBinghamDistribution(array([[1.0, 1j], [0.0, 1.0]]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_log_norm_const_finite(self):
        """log_norm_const must be finite."""
        import math

        self.assertTrue(math.isfinite(self.cB2.log_norm_const))
        self.assertTrue(math.isfinite(self.cB3.log_norm_const))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_dim(self):
        self.assertEqual(self.cB2.complex_dim, 2)
        self.assertEqual(self.cB3.complex_dim, 3)
        # The real manifold dimension is 2*d - 1
        self.assertEqual(self.cB2.dim, 3)
        self.assertEqual(self.cB3.dim, 5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_normalises_to_one_2d(self):
        """MC check: 2-D pdf integrates to 1 over S^3."""
        random.seed(12345)
        # Sample uniformly from S^3 using complex Gaussian projection
        real_part = random.normal(size=(2, 200_000))
        imag_part = random.normal(size=(2, 200_000))
        raw = real_part + 1j * imag_part
        norms = linalg.norm(raw, axis=0, keepdims=True)
        Z = raw / norms
        area = 2.0 * float(pi) ** 2  # surface area of S^3
        mc_integral = float(mean(self.cB2.pdf(Z))) * area
        npt.assert_almost_equal(mc_integral, 1.0, decimal=2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_positive(self):
        """pdf must return positive values."""
        z = array([1.0, 0.0], dtype=complex)
        self.assertGreater(self.cB2.pdf(z), 0.0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_batch_vs_single(self):
        """Vectorised pdf matches point-by-point evaluation."""
        random.seed(0)
        real_part = random.normal(size=(2, 10))
        imag_part = random.normal(size=(2, 10))
        raw = real_part + 1j * imag_part
        Z = raw / linalg.norm(raw, axis=0, keepdims=True)
        batch = self.cB2.pdf(Z)
        single = array([self.cB2.pdf(Z[:, k]) for k in range(10)])
        npt.assert_allclose(batch, single, rtol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_invariant_to_phase(self):
        """pdf(exp(i*alpha)*z) == pdf(z) for any global phase alpha."""
        z = array([0.6 + 0.3j, 0.0], dtype=complex)
        z[1] = sqrt(1 - abs(z[0]) ** 2)
        p0 = self.cB2.pdf(z)
        for alpha in [0.3, 1.0, float(pi)]:
            p1 = self.cB2.pdf(exp(1j * alpha) * z)
            npt.assert_almost_equal(p1, p0, decimal=10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_sample_shape(self):
        """sample returns the right shape."""
        random.seed(42)
        S = self.cB2.sample(50)
        self.assertEqual(S.shape, (2, 50))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_sample_unit_norm(self):
        """All samples must lie on the unit sphere."""
        random.seed(42)
        S = self.cB2.sample(100)
        norms = linalg.norm(S, axis=0)
        npt.assert_allclose(norms, ones(100), atol=1e-12)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_sample_3d_unit_norm(self):
        """3-D samples also lie on the unit sphere."""
        random.seed(7)
        S = self.cB3.sample(50)
        norms = linalg.norm(S, axis=0)
        npt.assert_allclose(norms, ones(50), atol=1e-12)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_log_norm_2d_analytic(self):
        a = 3.0
        B = diag(array([-a, 0.0], dtype=complex))
        log_C_expected = float(log(2 * pi**2 / a * (1 - exp(-a))))
        log_norm_got = ComplexBinghamDistribution.log_norm(B)
        npt.assert_almost_equal(-log_norm_got, log_C_expected, decimal=6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_log_norm_equal_eigenvalues(self):
        """Equal eigenvalues (uniform) should not raise."""
        from pyrecest.backend import zeros

        B = zeros((3, 3), dtype=complex)
        log_norm = ComplexBinghamDistribution.log_norm(B)
        import math

        self.assertTrue(math.isfinite(log_norm))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_fit_returns_instance(self):
        """fit() returns a ComplexBinghamDistribution instance."""
        random.seed(0)
        Z = self.cB2.sample(200)
        cB_fit = ComplexBinghamDistribution.fit(Z)
        self.assertIsInstance(cB_fit, ComplexBinghamDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_fit_recovers_eigenvalue_gap(self):
        """Fit recovers the eigenvalue gap (up to additive const)."""
        random.seed(0)
        B = diag(array([-10.0, 0.0], dtype=complex))
        cB = ComplexBinghamDistribution(B)
        Z = cB.sample(2000)
        cB_fit = ComplexBinghamDistribution.fit(Z)
        evals_fit = sort(real(linalg.eigvalsh(cB_fit.B)))
        evals_true = sort(real(linalg.eigvalsh(B)))
        gap_true = float(evals_true[-1] - evals_true[0])
        gap_fit = float(evals_fit[-1] - evals_fit[0])
        npt.assert_almost_equal(gap_fit, gap_true, decimal=0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_cauchy_schwarz_zero_for_identical(self):
        """D_CS(p, p) should be 0."""
        d = ComplexBinghamDistribution.cauchy_schwarz_divergence(self.cB2, self.cB2)
        npt.assert_almost_equal(d, 0.0, decimal=6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_cauchy_schwarz_symmetric(self):
        """D_CS(p, q) == D_CS(q, p)."""
        B3a = diag(array([-5.0, -1.0, 0.0], dtype=complex))
        B3b = diag(array([-3.0, -2.0, 0.0], dtype=complex))
        d_ab = ComplexBinghamDistribution.cauchy_schwarz_divergence(B3a, B3b)
        d_ba = ComplexBinghamDistribution.cauchy_schwarz_divergence(B3b, B3a)
        npt.assert_almost_equal(d_ab, d_ba, decimal=6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_cauchy_schwarz_nonneg(self):
        """Cauchy-Schwarz divergence must be >= 0."""
        B3a = diag(array([-5.0, -1.0, 0.0], dtype=complex))
        B3b = diag(array([-3.0, -2.0, 0.0], dtype=complex))
        d = ComplexBinghamDistribution.cauchy_schwarz_divergence(B3a, B3b)
        self.assertGreaterEqual(d, -1e-10)


if __name__ == "__main__":
    unittest.main()
