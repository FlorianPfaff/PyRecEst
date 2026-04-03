import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, pi
from pyrecest.distributions.circle.sine_skewed_distributions import (
    GeneralizedKSineSkewedVonMisesDistribution,
    GeneralizedKSineSkewedWrappedCauchyDistribution,
    SineSkewedWrappedCauchyDistribution,
    SineSkewedWrappedNormalDistribution,
)


class TestGeneralizedKSineSkewedVonMisesDistribution(unittest.TestCase):
    def test_initialization(self):
        """Test initialization with valid and invalid parameters."""
        # Test with valid parameters
        try:
            GeneralizedKSineSkewedVonMisesDistribution(
                mu=pi, kappa=1, lambda_=0.5, k=1, m=1
            )
        except NotImplementedError as e:
            self.fail(f"Initialization with valid parameters failed: {e}")

        # Test with invalid lambda_
        with self.assertRaises(AssertionError):
            GeneralizedKSineSkewedVonMisesDistribution(
                mu=pi, kappa=1, lambda_=1.5, k=1, m=1
            )

        # Test with invalid m
        with self.assertRaises(AssertionError):
            GeneralizedKSineSkewedVonMisesDistribution(
                mu=pi, kappa=1, lambda_=0.5, k=1, m=0
            )

    def test_pdf(self):
        """Test the pdf method for expected behavior."""
        dist = GeneralizedKSineSkewedVonMisesDistribution(
            mu=pi, kappa=1, lambda_=0.5, k=1, m=1
        )
        xa = pi / 2
        pdf_val = dist.pdf(xa)
        # This is more of a sanity check; a more comprehensive test would compare against a known result
        self.assertTrue(0 <= pdf_val <= 2, "PDF value out of expected range.")

    def test_shift(self):
        """Test the shift method modifies mu correctly."""
        dist = GeneralizedKSineSkewedVonMisesDistribution(
            mu=0, kappa=1, lambda_=0.5, k=1, m=1
        )
        new_dist = dist.shift(array(pi / 2))
        self.assertEqual(
            new_dist.mu, pi / 2, "Shift method did not update mu correctly."
        )


def test_sine_skewed_wrapped_normal_initialization():
    mu = array(0.0)
    sigma = array(1.0)
    lambda_ = array(0.5)
    dist = SineSkewedWrappedNormalDistribution(mu, sigma, lambda_)
    assert dist.mu == mu
    assert dist.sigma == sigma
    assert dist.lambda_ == lambda_


def test_sine_skewed_wrapped_cauchy_initialization():
    mu = array(pi / 4)
    gamma = array(0.1)
    lambda_ = array(-0.5)
    dist = SineSkewedWrappedCauchyDistribution(mu, gamma, lambda_)
    assert dist.mu == mu
    assert dist.gamma == gamma
    assert dist.lambda_ == lambda_


def test_sine_skewed_wrapped_normal_pdf():
    mu = array(0.0)
    sigma = array(1.0)
    lambda_ = array(0.0)
    xs = array([0.0, pi / 2, pi, 3 * pi / 2])
    dist = SineSkewedWrappedNormalDistribution(mu, sigma, lambda_)
    pdf_values = dist.pdf(xs)
    assert len(pdf_values) == len(xs)
    assert all(pdf_values >= 0)  # PDF values should be non-negative


def test_sine_skewed_wrapped_cauchy_pdf():
    mu = array(pi / 4)
    gamma = array(0.1)
    lambda_ = array(0.0)
    xs = array([0.0, pi / 2, pi, 3 * pi / 2])
    dist = SineSkewedWrappedCauchyDistribution(mu, gamma, lambda_)
    pdf_values = dist.pdf(xs)
    assert len(pdf_values) == len(xs)
    assert all(pdf_values >= 0)  # PDF values should be non-negative


def test_sine_skewed_effect():
    mu = array(0.0)
    sigma = array(1.0)
    lambda_ = array(1.0)  # Max skew
    normal_dist = SineSkewedWrappedNormalDistribution(mu, sigma, 0)  # No skew
    skewed_dist = SineSkewedWrappedNormalDistribution(mu, sigma, lambda_)  # Max skew
    assert skewed_dist.pdf(mu + 0.1) > normal_dist.pdf(mu)
    assert skewed_dist.pdf(mu - 0.1) < normal_dist.pdf(mu)

    lambda_ = -1  # Negative skew
    skewed_dist = SineSkewedWrappedNormalDistribution(mu, sigma, lambda_)
    assert skewed_dist.pdf(mu + 0.1) < normal_dist.pdf(mu)
    assert skewed_dist.pdf(mu - 0.1) > normal_dist.pdf(mu)


class TestGeneralizedKSineSkewedWrappedCauchyDistribution(unittest.TestCase):
    def test_initialization(self):
        """Test initialization with valid and invalid parameters."""
        GeneralizedKSineSkewedWrappedCauchyDistribution(
            mu=pi, gamma=0.5, lambda_=0.5, k=1, m=1
        )

        with self.assertRaises(AssertionError):
            GeneralizedKSineSkewedWrappedCauchyDistribution(
                mu=pi, gamma=0.5, lambda_=1.5, k=1, m=1
            )

        with self.assertRaises(AssertionError):
            GeneralizedKSineSkewedWrappedCauchyDistribution(
                mu=pi, gamma=0.5, lambda_=0.5, k=1, m=0
            )

        with self.assertRaises(AssertionError):
            GeneralizedKSineSkewedWrappedCauchyDistribution(
                mu=pi, gamma=-0.1, lambda_=0.5, k=1, m=1
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("pytorch", "jax"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_pdf_m1_normalizes(self):
        """m=1 GSSC should integrate to 1."""
        dist = GeneralizedKSineSkewedWrappedCauchyDistribution(
            mu=pi / 3, gamma=0.5, lambda_=0.4, k=1, m=1
        )
        integral = dist.integrate_numerically()

        npt.assert_allclose(integral, 1.0, atol=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("pytorch", "jax"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_pdf_m2_normalizes(self):
        """Test that m=2 PDF integrates to approximately 1."""
        dist = GeneralizedKSineSkewedWrappedCauchyDistribution(
            mu=pi / 4, gamma=0.3, lambda_=0.5, k=1, m=2
        )
        integral = dist.integrate_numerically()

        npt.assert_allclose(integral, 1.0, atol=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("pytorch", "jax"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_pdf_m3_normalizes(self):
        """Test that m=3 PDF integrates to approximately 1."""
        dist = GeneralizedKSineSkewedWrappedCauchyDistribution(
            mu=0.0, gamma=0.5, lambda_=0.3, k=1, m=3
        )
        integral = dist.integrate_numerically()

        npt.assert_allclose(integral, 1.0, atol=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("pytorch", "jax"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_pdf_m4_normalizes(self):
        """Test that m=4 PDF integrates to approximately 1."""
        dist = GeneralizedKSineSkewedWrappedCauchyDistribution(
            mu=pi, gamma=0.4, lambda_=0.6, k=1, m=4
        )
        integral = dist.integrate_numerically()

        npt.assert_allclose(integral, 1.0, atol=1e-4)

    def test_pdf_nonnegative(self):
        """Test that PDF values are non-negative."""
        for m in (1, 2, 3, 4):
            dist = GeneralizedKSineSkewedWrappedCauchyDistribution(
                mu=pi / 2, gamma=0.3, lambda_=0.5, k=1, m=m
            )
            xs = array([0.0, pi / 4, pi / 2, pi, 3 * pi / 2, 2 * pi - 0.01])
            assert all(dist.pdf(xs) >= 0), f"Negative PDF value for m={m}"

    def test_shift(self):
        """Test the shift method modifies mu correctly."""
        dist = GeneralizedKSineSkewedWrappedCauchyDistribution(
            mu=0, gamma=0.5, lambda_=0.4, k=1, m=1
        )
        new_dist = dist.shift(array(pi / 2))

        npt.assert_allclose(float(new_dist.mu), pi / 2, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
