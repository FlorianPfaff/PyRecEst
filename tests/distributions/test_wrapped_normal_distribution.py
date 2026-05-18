import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    allclose,
    arange,
    array,
    exp,
    isfinite,
    ones_like,
    pi,
    sqrt,
    sum,
)
from pyrecest.distributions import WrappedNormalDistribution


class WrappedNormalDistributionTest(unittest.TestCase):
    def setUp(self):
        self.mu = array(3.0)
        self.sigma = array(1.5)
        self.wn = WrappedNormalDistribution(self.mu, self.sigma)

    def test_pdf_values_are_as_expected(self):
        """
        Test that the probability density function (pdf) returns expected values.
        """

        def approx_with_wrapping(x):
            k = arange(-20, 21)
            total = sum(exp(-((x - self.mu + 2 * pi * k) ** 2) / (2 * self.sigma**2)))
            return 1 / sqrt(2 * pi) / self.sigma * total

        test_points = [self.mu, self.mu - 1, self.mu + 2]
        for point in test_points:
            with self.subTest(x=point):
                npt.assert_almost_equal(
                    self.wn.pdf(point), approx_with_wrapping(point), decimal=7
                )

        x = arange(0, 7)
        self.assertTrue(
            allclose(
                self.wn.pdf(x),
                array([approx_with_wrapping(xi) for xi in x]),
                rtol=1e-7,
            )
        )

    def test_pdf_is_periodic_when_central_term_underflows(self):
        """The pdf must still add wrapped terms near the 0 / 2π boundary."""
        wn = WrappedNormalDistribution(array(0.0), array(0.1))

        density_near_left_boundary = wn.pdf(array(0.1))
        density_near_right_boundary = wn.pdf(array(2.0 * pi - 0.1))

        self.assertTrue(
            allclose(density_near_right_boundary, density_near_left_boundary, rtol=1e-5)
        )

    def test_pdf_with_large_sigma_is_uniform(self):
        """
        Test that the pdf with large sigma is approximately a uniform distribution.
        """
        wn_large_sigma = WrappedNormalDistribution(array(0.0), array(100.0))
        x = arange(0, 7)
        fx = ones_like(x) / (2.0 * pi)
        self.assertTrue(allclose(wn_large_sigma.pdf(x), fx, rtol=1e-10))

    @unittest.skipUnless(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Regression test for the JAX-specific wrapped-normal PDF loop.",
    )
    def test_jax_pdf_stops_after_increment_converges(self):
        """The JAX PDF loop should stop based on the latest wrap increment.

        A previous convergence check inspected the accumulated positive density
        itself, so the JAX branch ran until max_iterations for ordinary inputs.
        """
        import jax  # pylint: disable=import-error,import-outside-toplevel

        original_while_loop = jax.lax.while_loop
        iterations = {"count": 0}

        def counting_while_loop(cond_fun, body_fun, init_val):
            val = init_val
            while bool(cond_fun(val)):
                iterations["count"] += 1
                val = body_fun(val)
            return val

        jax.lax.while_loop = counting_while_loop
        try:
            dist = WrappedNormalDistribution(array(0.0), array(0.7))
            value = dist.pdf(array([0.0]))
        finally:
            jax.lax.while_loop = original_while_loop

        self.assertTrue(bool(isfinite(value)))
        self.assertLess(iterations["count"], 10)

    def test_multiply_uses_explicit_vm_approximation(self):
        """The product API is a documented VM-based approximation."""
        first = WrappedNormalDistribution(array(0.0), array(0.5))
        second = WrappedNormalDistribution(array(3.0), array(0.6))

        multiplied = first.multiply(second)
        explicit = first.multiply_vm_approximation(second)
        legacy = first.multiply_vm(second)

        self.assertTrue(allclose(multiplied.mu, explicit.mu, atol=1e-12))
        self.assertTrue(allclose(multiplied.sigma, explicit.sigma, atol=1e-12))
        self.assertTrue(allclose(legacy.mu, explicit.mu, atol=1e-12))
        self.assertTrue(allclose(legacy.sigma, explicit.sigma, atol=1e-12))

    def test_multiply_is_not_treated_as_exact_density_product(self):
        """A single wrapped normal is not exact product-closed on the circle."""
        first = WrappedNormalDistribution(array(0.0), array(0.5))
        second = WrappedNormalDistribution(array(3.0), array(0.6))
        xs = arange(0, 256) / 256 * 2.0 * pi
        dx = 2.0 * pi / 256

        product_pdf = first.pdf(xs) * second.pdf(xs)
        normalized_product_pdf = product_pdf / (sum(product_pdf) * dx)
        approximate_product_pdf = first.multiply(second).pdf(xs)

        self.assertFalse(
            allclose(
                approximate_product_pdf,
                normalized_product_pdf,
                atol=0.1,
                rtol=0.0,
            )
        )

    def test_convolve_adds_variances(self):
        """Convolution must add wrapped-normal variances, not square them."""
        first = WrappedNormalDistribution(array(0.3), array(2.0))
        second = WrappedNormalDistribution(array(0.4), array(3.0))

        convolved = first.convolve(second)

        self.assertTrue(allclose(convolved.mu, array(0.7), atol=1e-12))
        self.assertTrue(allclose(convolved.C, array(13.0), atol=1e-12))
        self.assertTrue(allclose(convolved.sigma, sqrt(array(13.0)), atol=1e-12))


if __name__ == "__main__":
    unittest.main()
