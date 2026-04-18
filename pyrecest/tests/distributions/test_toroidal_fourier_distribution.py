import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    array,
    column_stack,
    diag,
    linspace,
    meshgrid,
    pi,
    sum,
    zeros,
)
from pyrecest.distributions import (
    HypertoroidalFourierDistribution,
    ToroidalWrappedNormalDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_fourier_distribution import (
    ToroidalFourierDistribution,
)


def integrate2d(dist, N=100):
    """Numerically integrate a 2D toroidal distribution over [0,2pi)^2."""
    x = linspace(0, 2 * pi, N, endpoint=False) + (pi / N)
    y = linspace(0, 2 * pi, N, endpoint=False) + (pi / N)
    X, Y = meshgrid(x, y, indexing="ij")
    pts = column_stack((X.flatten(), Y.flatten()))
    p = dist.pdf(pts)
    area = (2 * pi / N) ** 2
    return float(sum(p) * area)


class TestToroidalFourierDistributionConstructor(unittest.TestCase):
    def test_basic_construction_sqrt(self):
        twn = ToroidalWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([0.3, 0.5]))
        )
        tfd = ToroidalFourierDistribution.from_distribution(twn, (11, 11), "sqrt")
        self.assertIsInstance(tfd, ToroidalFourierDistribution)
        self.assertEqual(tfd.dim, 2)
        self.assertEqual(tfd.transformation, "sqrt")

    def test_basic_construction_identity(self):
        twn = ToroidalWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([0.3, 0.5]))
        )
        tfd = ToroidalFourierDistribution.from_distribution(twn, (11, 11), "identity")
        self.assertIsInstance(tfd, ToroidalFourierDistribution)
        self.assertEqual(tfd.transformation, "identity")

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_vector_input_raises(self):
        c = zeros(5, dtype=complex)
        c[2] = 1.0 / (2 * pi)
        with self.assertRaises(ValueError):
            ToroidalFourierDistribution(array(c), "identity")

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_from_distribution_returns_correct_type(self):
        twn = ToroidalWrappedNormalDistribution(
            array([0.5, 1.5]), diag(array([0.2, 0.4]))
        )
        tfd = ToroidalFourierDistribution.from_distribution(twn, (9, 9), "sqrt")
        self.assertIsInstance(tfd, ToroidalFourierDistribution)
        self.assertIsInstance(tfd, HypertoroidalFourierDistribution)

    def test_from_function_values_returns_correct_type(self):
        twn = ToroidalWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([0.3, 0.5]))
        )
        n = 15
        x = linspace(0, 2 * pi, n, endpoint=False)
        y = linspace(0, 2 * pi, n, endpoint=False)
        X, Y = meshgrid(x, y, indexing="ij")
        pts = column_stack((X.flatten(), Y.flatten()))
        fvals = twn.pdf(pts).reshape((n, n))
        tfd = ToroidalFourierDistribution.from_function_values(
            fvals, (n, n), "identity"
        )
        self.assertIsInstance(tfd, ToroidalFourierDistribution)


class TestToroidalFourierDistributionPDF(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_values_close_to_twn_sqrt(self):
        twn = ToroidalWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([0.3, 0.5]))
        )
        tfd = ToroidalFourierDistribution.from_distribution(twn, (17, 17), "sqrt")
        pts = array([[1.0, 2.0], [0.5, 1.5], [2.0, 4.0]])
        npt.assert_allclose(tfd.pdf(pts), twn.pdf(pts), atol=1e-3)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_values_close_to_twn_identity(self):
        twn = ToroidalWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([0.3, 0.5]))
        )
        tfd = ToroidalFourierDistribution.from_distribution(twn, (17, 17), "identity")
        pts = array([[1.0, 2.0], [0.5, 1.5], [2.0, 4.0]])
        npt.assert_allclose(tfd.pdf(pts), twn.pdf(pts), atol=1e-3)


class TestToroidalFourierDistributionIntegrate(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("pytorch", "jax"),  # pylint: disable=no-member
        reason="Not supported on pytorch or JAX backend",
    )
    def test_integrate_full_domain_sqrt(self):
        twn = ToroidalWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([0.3, 0.5]))
        )
        tfd = ToroidalFourierDistribution.from_distribution(twn, (15, 15), "sqrt")
        result = tfd.integrate()
        self.assertAlmostEqual(result, 1.0, places=4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_integrate_full_domain_identity(self):
        twn = ToroidalWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([0.3, 0.5]))
        )
        tfd = ToroidalFourierDistribution.from_distribution(twn, (15, 15), "identity")
        result = tfd.integrate()
        self.assertAlmostEqual(result, 1.0, places=4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_integrate_partial_domain(self):
        twn = ToroidalWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([0.3, 0.5]))
        )
        tfd = ToroidalFourierDistribution.from_distribution(twn, (17, 17), "identity")
        # The two complementary half-domains must partition the full domain
        bounds1 = array([[0.0, pi], [0.0, 2 * pi]])
        bounds2 = array([[pi, 2 * pi], [0.0, 2 * pi]])
        r1 = tfd.integrate(bounds1)
        r2 = tfd.integrate(bounds2)
        self.assertAlmostEqual(r1 + r2, 1.0, places=4)
        # Each partial integral must be positive
        self.assertGreater(r1, 0.0)
        self.assertGreater(r2, 0.0)


class TestToroidalFourierDistributionMoments(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("pytorch", "jax"),  # pylint: disable=no-member
        reason="Not supported on pytorch or JAX backend",
    )
    def test_mean_direction_close_to_twn(self):
        mu = array([1.0, 2.5])
        twn = ToroidalWrappedNormalDistribution(mu, diag(array([0.2, 0.3])))
        tfd = ToroidalFourierDistribution.from_distribution(twn, (13, 13), "sqrt")
        npt.assert_allclose(tfd.mean_direction(), mu, atol=1e-2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_trigonometric_moment_n0(self):
        twn = ToroidalWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([0.3, 0.5]))
        )
        tfd = ToroidalFourierDistribution.from_distribution(twn, (11, 11), "sqrt")
        m0 = tfd.trigonometric_moment(0)
        npt.assert_allclose(m0.real, array([1.0, 1.0]), atol=1e-10)


class TestToroidalFourierDistributionToTWN(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("pytorch", "jax"),  # pylint: disable=no-member
        reason="Not supported on pytorch or JAX backend",
    )
    def test_to_twn_recovers_mean(self):
        mu = array([1.0, 2.5])
        twn = ToroidalWrappedNormalDistribution(mu, diag(array([0.2, 0.3])))
        tfd = ToroidalFourierDistribution.from_distribution(twn, (21, 21), "sqrt")
        twn_recovered = tfd.to_twn()
        npt.assert_allclose(twn_recovered.mu, mu, atol=1e-2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("pytorch", "jax"),  # pylint: disable=no-member
        reason="Not supported on pytorch or JAX backend",
    )
    def test_to_twn_recovers_covariance(self):
        mu = array([1.0, 2.5])
        C = diag(array([0.2, 0.3]))
        twn = ToroidalWrappedNormalDistribution(mu, C)
        tfd = ToroidalFourierDistribution.from_distribution(twn, (21, 21), "sqrt")
        twn_recovered = tfd.to_twn()
        npt.assert_allclose(twn_recovered.C[0, 0], C[0, 0], atol=5e-2)
        npt.assert_allclose(twn_recovered.C[1, 1], C[1, 1], atol=5e-2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("pytorch", "jax"),  # pylint: disable=no-member
        reason="Not supported on pytorch or JAX backend",
    )
    def test_to_twn_returns_twn_type(self):
        twn = ToroidalWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([0.3, 0.5]))
        )
        tfd = ToroidalFourierDistribution.from_distribution(twn, (15, 15), "sqrt")
        twn_out = tfd.to_twn()
        self.assertIsInstance(twn_out, ToroidalWrappedNormalDistribution)


class TestToroidalFourierDistributionCircularCorrelation(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("pytorch", "jax"),  # pylint: disable=no-member
        reason="Not supported on pytorch or JAX backend",
    )
    def test_circular_correlation_independent(self):
        twn = ToroidalWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([0.3, 0.5]))
        )
        tfd = ToroidalFourierDistribution.from_distribution(twn, (15, 15), "sqrt")
        rhoc = tfd.circular_correlation_jammalamadaka()
        self.assertAlmostEqual(rhoc, 0.0, delta=0.05)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_circular_correlation_matches_numerical(self):
        twn = ToroidalWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([0.3, 0.5]))
        )
        tfd = ToroidalFourierDistribution.from_distribution(twn, (17, 17), "sqrt")
        rhoc_analytical = tfd.circular_correlation_jammalamadaka()
        rhoc_numerical = tfd.circular_correlation_jammalamadaka_numerical()
        self.assertAlmostEqual(rhoc_analytical, rhoc_numerical, delta=0.05)


if __name__ == "__main__":
    unittest.main()
