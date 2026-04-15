import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import (
    array,
    complex128,
    conj,
    eye,
    ones,
    pi,
    real,
    sqrt,
    sum,
    trace,
)
from pyrecest.distributions import ComplexAngularCentralGaussianDistribution


class TestComplexAngularCentralGaussianDistribution(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Identity matrix case (uniform distribution on complex unit sphere)
        self.C_identity_2d = eye(2, dtype=complex128)
        self.dist_identity_2d = ComplexAngularCentralGaussianDistribution(
            self.C_identity_2d
        )

        # Non-trivial Hermitian positive definite matrix for 2D case
        # C = [[2, 1+1j], [1-1j, 3]]
        self.C_nontrivial_2d = array([[2.0, 1.0 + 1.0j], [1.0 - 1.0j, 3.0]])
        self.dist_nontrivial_2d = ComplexAngularCentralGaussianDistribution(
            self.C_nontrivial_2d
        )

    def test_constructor_valid(self):
        """Test that constructor accepts a Hermitian matrix."""
        self.assertEqual(self.dist_identity_2d.dim, 2)
        self.assertEqual(self.dist_nontrivial_2d.dim, 2)

    def test_constructor_non_hermitian_raises(self):
        """Test that constructor rejects a non-Hermitian matrix."""
        C_bad = array([[1.0 + 0j, 2.0 + 1.0j], [0.0 + 0j, 1.0 + 0j]])
        with self.assertRaises(AssertionError):
            ComplexAngularCentralGaussianDistribution(C_bad)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_pdf_identity_uniform(self):
        """For C=I, the pdf should be constant gamma(d)/(2*pi^d) on the unit sphere."""
        d = 2
        # Expected: gamma(2) / (2*pi^2) = 1 / (2*pi^2)
        expected = 1.0 / (2.0 * float(pi) ** d)  # gamma(2)=1

        # Test on several unit vectors
        inv_sqrt2 = 1.0 / float(sqrt(array(2.0)))
        z1 = array([[1.0 + 0j, 0.0 + 0j]])
        z2 = array([[0.0 + 0j, 1.0 + 0j]])
        z3 = array([[inv_sqrt2 + 0j, 1j * inv_sqrt2]])
        z4 = array([[0.5 + 0.5j, 0.5 - 0.5j]])

        for z in [z1, z2, z3, z4]:
            p = self.dist_identity_2d.pdf(z)
            npt.assert_allclose(
                float(real(p[0])),
                expected,
                rtol=1e-6,
                err_msg=f"PDF for identity C is not constant at {z}",
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_pdf_positive(self):
        """PDF values should be positive for any unit vector."""
        inv_sqrt2 = 1.0 / float(sqrt(array(2.0)))
        z = array([[inv_sqrt2 + 0j, 1j * inv_sqrt2]])
        p = self.dist_nontrivial_2d.pdf(z)
        self.assertGreater(float(real(p[0])), 0.0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_pdf_batch_vs_single(self):
        """Batch PDF evaluation should match individual evaluations."""
        inv_sqrt2 = 1.0 / float(sqrt(array(2.0)))
        z_list = [
            array([[1.0 + 0j, 0.0 + 0j]]),
            array([[0.0 + 0j, 1.0 + 0j]]),
            array([[inv_sqrt2 + 0j, 1j * inv_sqrt2]]),
        ]
        za = array(
            [
                [1.0 + 0j, 0.0 + 0j],
                [0.0 + 0j, 1.0 + 0j],
                [inv_sqrt2 + 0j, 1j * inv_sqrt2],
            ]
        )

        p_batch = self.dist_nontrivial_2d.pdf(za)
        for i, z in enumerate(z_list):
            p_single = self.dist_nontrivial_2d.pdf(z)
            npt.assert_allclose(
                float(real(p_batch[i])),
                float(real(p_single[0])),
                rtol=1e-10,
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_sample_unit_norm(self):
        """Sampled vectors should lie on the complex unit sphere."""
        n = 100
        Z = self.dist_nontrivial_2d.sample(n)
        norms_sq = array([float(real(sum(Z[k] * conj(Z[k])))) for k in range(n)])
        npt.assert_allclose(norms_sq, ones(n), atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_sample_correct_dim(self):
        """Sampled vectors should have the correct shape."""
        n = 50
        Z = self.dist_identity_2d.sample(n)
        self.assertEqual(Z.shape[0], n)
        self.assertEqual(Z.shape[1], 2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_estimate_parameter_matrix_identity(self):
        """Fitting samples from identity-C distribution should recover approx identity."""
        pyrecest.backend.random.seed(42)  # pylint: disable=no-member
        n = 2000
        Z = self.dist_identity_2d.sample(n)
        C_est = ComplexAngularCentralGaussianDistribution.estimate_parameter_matrix(
            Z, n_iterations=100
        )
        # Normalize C_est to have trace equal to 2 (matching identity)
        C_est_normalized = C_est / real(trace(C_est)) * 2.0
        npt.assert_allclose(
            real(C_est_normalized),
            eye(2),
            atol=0.15,
            err_msg="Estimated C does not approximately match identity",
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_fit_returns_distribution(self):
        """fit() should return a ComplexAngularCentralGaussianDistribution."""
        pyrecest.backend.random.seed(0)  # pylint: disable=no-member
        Z = self.dist_identity_2d.sample(50)
        dist = ComplexAngularCentralGaussianDistribution.fit(Z, n_iterations=10)
        self.assertIsInstance(dist, ComplexAngularCentralGaussianDistribution)
        self.assertEqual(dist.dim, 2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_3d_case(self):
        """Test basic functionality for d=3."""
        dist = ComplexAngularCentralGaussianDistribution(eye(3, dtype=complex128))
        self.assertEqual(dist.dim, 3)

        Z = dist.sample(20)
        self.assertEqual(Z.shape, (20, 3))

        # Check unit norms
        norms_sq = array([float(real(sum(Z[k] * conj(Z[k])))) for k in range(20)])
        npt.assert_allclose(norms_sq, ones(20), atol=1e-10)

        # For d=3, C=I: pdf should be gamma(3)/(2*pi^3) = 2/(2*pi^3) = 1/pi^3
        z_test = array([[1.0 + 0j, 0.0 + 0j, 0.0 + 0j]])
        p = dist.pdf(z_test)
        expected = 1.0 / float(pi) ** 3  # gamma(3)=2, so 2/(2*pi^3)=1/pi^3
        npt.assert_allclose(float(real(p[0])), expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
