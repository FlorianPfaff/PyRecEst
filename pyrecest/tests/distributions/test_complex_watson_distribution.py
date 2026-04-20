# pylint: disable=no-name-in-module,no-member,redefined-builtin
import unittest

import pyrecest.backend
from pyrecest.backend import (
    all,
    allclose,
    array,
    complex128,
    conj,
    exp,
    isfinite,
    linalg,
    ones,
    real,
    sum,
)
from pyrecest.distributions import ComplexWatsonDistribution


class TestComplexWatsonDistribution(unittest.TestCase):
    def setUp(self):
        # 2-D complex unit vector
        self.mu2 = array([1.0, 0.0], dtype=complex128)
        self.kappa2 = 2.0

        # 3-D complex unit vector
        raw = array([1.0, 1j, 1.0 + 1j], dtype=complex128)
        self.mu3 = raw / linalg.norm(raw)
        self.kappa3 = 5.0

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def test_constructor_stores_attributes(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        self.assertTrue(bool(all(cw.mu == self.mu2)))
        self.assertEqual(cw.kappa, self.kappa2)
        self.assertEqual(cw.dim, 2)

    def test_constructor_rejects_unnormalized_mu(self):
        with self.assertRaises(AssertionError):
            ComplexWatsonDistribution(array([1.0, 1.0], dtype=complex128), 1.0)

    def test_constructor_rejects_2d_mu(self):
        with self.assertRaises(AssertionError):
            ComplexWatsonDistribution(array([[1.0, 0.0]], dtype=complex128), 1.0)

    # ------------------------------------------------------------------
    # mean
    # ------------------------------------------------------------------
    def test_mean_returns_mu(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        self.assertTrue(bool(all(cw.mean() == self.mu2)))

    # ------------------------------------------------------------------
    # log_norm
    # ------------------------------------------------------------------
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_log_norm_scalar(self):
        lc = ComplexWatsonDistribution.log_norm(2, 2.0)
        self.assertTrue(bool(isfinite(array(lc))))
        self.assertIsInstance(lc, float)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_log_norm_array(self):
        kappas = array([0.0, 0.1, 1.0, 10.0, 200.0])
        lcs = ComplexWatsonDistribution.log_norm(3, kappas)
        self.assertEqual(lcs.shape, kappas.shape)
        self.assertTrue(bool(all(isfinite(lcs))))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_log_norm_consistency_across_regimes(self):
        D = 3
        kappa_border = 1.0 / D
        delta = 1e-4
        lc_lo = ComplexWatsonDistribution.log_norm(D, kappa_border - delta)
        lc_me = ComplexWatsonDistribution.log_norm(D, kappa_border + delta)
        self.assertAlmostEqual(lc_lo, lc_me, places=2)

        lc_me2 = ComplexWatsonDistribution.log_norm(D, 100.0 - 1e-6)
        lc_hi = ComplexWatsonDistribution.log_norm(D, 100.0 + 1e-6)
        self.assertAlmostEqual(lc_me2, lc_hi, places=2)

    # ------------------------------------------------------------------
    # pdf
    # ------------------------------------------------------------------
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_nonnegative(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        pts = array([[1.0, 0.0], [0.0, 1.0], [1j, 0.0]], dtype=complex128)
        pts = pts / linalg.norm(pts, axis=1, keepdims=True)
        p = cw.pdf(pts)
        self.assertTrue(bool(all(p >= 0.0)))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_mode_is_maximum(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        p_mode = cw.pdf(self.mu2)
        other = array([0.0, 1.0], dtype=complex128)
        p_other = cw.pdf(other)
        self.assertGreaterEqual(float(p_mode), float(p_other))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_antipodal_symmetry(self):
        cw = ComplexWatsonDistribution(self.mu3, self.kappa3)
        raw = array([1.0, -1j, 0.5], dtype=complex128)
        z = raw / linalg.norm(raw)
        self.assertTrue(bool(allclose(cw.pdf(z), cw.pdf(-z), rtol=1e-12)))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_phase_invariance(self):
        cw = ComplexWatsonDistribution(self.mu3, self.kappa3)
        raw = array([1.0, 0.5, -0.5j], dtype=complex128)
        z = raw / linalg.norm(raw)
        phase = exp(array(1j * 0.7, dtype=complex128))
        self.assertTrue(bool(allclose(cw.pdf(z), cw.pdf(phase * z), rtol=1e-12)))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_single_point(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        p = cw.pdf(self.mu2)
        self.assertIsInstance(float(p), float)

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_sample_shape(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        samples = cw.sample(50)
        self.assertEqual(samples.shape, (50, 2))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_sample_unit_norm(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        samples = cw.sample(200)
        norms = linalg.norm(samples, axis=1)
        self.assertTrue(bool(allclose(norms, ones(200), atol=1e-10)))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_sample_3d_unit_norm(self):
        cw = ComplexWatsonDistribution(self.mu3, self.kappa3)
        samples = cw.sample(100)
        norms = linalg.norm(samples, axis=1)
        self.assertTrue(bool(allclose(norms, ones(100), atol=1e-10)))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_sample_raises_for_dim1(self):
        mu1 = array([1.0 + 0j], dtype=complex128)
        with self.assertRaises(ValueError):
            ComplexWatsonDistribution(mu1, 2.0).sample(10)

    # ------------------------------------------------------------------
    # estimate_parameters / fit
    # ------------------------------------------------------------------
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_estimate_parameters_recovers_mu(self):
        cw_true = ComplexWatsonDistribution(self.mu2, self.kappa2)
        Z = cw_true.sample(500)
        mu_hat, _ = ComplexWatsonDistribution.estimate_parameters(Z)
        overlap = abs(float(real(sum(conj(mu_hat) * self.mu2))))
        self.assertGreater(overlap, 0.9)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_estimate_parameters_recovers_kappa(self):
        cw_true = ComplexWatsonDistribution(self.mu2, self.kappa2)
        Z = cw_true.sample(1000)
        _, kappa_hat = ComplexWatsonDistribution.estimate_parameters(Z)
        self.assertAlmostEqual(kappa_hat, self.kappa2, delta=1.0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_fit_returns_distribution(self):
        cw_true = ComplexWatsonDistribution(self.mu2, self.kappa2)
        Z = cw_true.sample(200)
        cw_fit = ComplexWatsonDistribution.fit(Z)
        self.assertIsInstance(cw_fit, ComplexWatsonDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_fit_with_weights(self):
        cw_true = ComplexWatsonDistribution(self.mu2, self.kappa2)
        Z = cw_true.sample(200)
        w = ones(200)
        cw_fit = ComplexWatsonDistribution.fit(Z, weights=w)
        self.assertIsInstance(cw_fit, ComplexWatsonDistribution)


if __name__ == "__main__":
    unittest.main()
