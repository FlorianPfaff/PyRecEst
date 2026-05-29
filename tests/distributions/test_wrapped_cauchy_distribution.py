import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arange, array, pi
from pyrecest.distributions.circle.custom_circular_distribution import (
    CustomCircularDistribution,
)
from pyrecest.distributions.circle.wrapped_cauchy_distribution import (
    WrappedCauchyDistribution,
)


class WrappedCauchyDistributionTest(unittest.TestCase):
    def setUp(self):
        self.mu = 0.0
        self.gamma = 0.5
        self.xs = arange(10)

    def test_pdf(self):
        dist = WrappedCauchyDistribution(self.mu, self.gamma)

        def pdf_wrapped(x, mu, gamma, terms=2000):
            summation = 0
            for k in range(-terms, terms + 1):
                summation += gamma / (pi * (gamma**2 + (x - mu + 2.0 * pi * k) ** 2))
            return summation

        custom_wrapped = CustomCircularDistribution(
            lambda xs: array([pdf_wrapped(x, self.mu, self.gamma) for x in xs])
        )

        npt.assert_allclose(
            dist.pdf(xs=self.xs), custom_wrapped.pdf(xs=self.xs), atol=0.0001
        )

    def test_pdf_mode_for_nonzero_mean(self):
        dist = WrappedCauchyDistribution(array(1.0), array(0.5))
        npt.assert_array_less(dist.pdf(array([2.0])), dist.pdf(array([1.0])))

    def test_pdf_accepts_list_inputs(self):
        dist = WrappedCauchyDistribution(self.mu, self.gamma)
        xs = [0.1, 0.2, 0.3]

        npt.assert_allclose(dist.pdf(xs), dist.pdf(array(xs)))

    def test_pdf_accepts_scalar_inputs(self):
        dist = WrappedCauchyDistribution(self.mu, self.gamma)

        npt.assert_allclose(dist.pdf(0.5), dist.pdf(array([0.5])))
        npt.assert_allclose(dist.pdf(array(0.5)), dist.pdf(array([0.5])))

    def test_rejects_invalid_gamma(self):
        for gamma in (0.0, -0.5, float("inf"), array([0.5, 1.0])):
            with self.subTest(gamma=gamma):
                with self.assertRaisesRegex(ValueError, "gamma"):
                    WrappedCauchyDistribution(self.mu, gamma)

    def test_pdf_rejects_matrix_inputs(self):
        dist = WrappedCauchyDistribution(self.mu, self.gamma)

        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            dist.pdf(array([[0.1, 0.2]]))

    def test_cdf_rejects_matrix_inputs(self):
        dist = WrappedCauchyDistribution(self.mu, self.gamma)

        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            dist.cdf(array([[0.1, 0.2]]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_cdf(self):
        dist = WrappedCauchyDistribution(self.mu, self.gamma)
        npt.assert_allclose(dist.cdf(array([1.0])), dist.integrate(array([0.0, 1.0])))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_cdf_across_arctan_branch_cut(self):
        dist = WrappedCauchyDistribution(self.mu, self.gamma)
        xs = array([pi - 1e-6, pi + 1e-6, 2.0 * pi - 1e-6])

        npt.assert_allclose(dist.cdf(xs), dist.cdf_numerical(xs), atol=1e-8)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_cdf_with_nonzero_mean(self):
        dist = WrappedCauchyDistribution(array(1.0), array(0.5))
        xs = array([0.5, 1.0, 2.0, 4.0])

        npt.assert_allclose(dist.cdf(xs), dist.cdf_numerical(xs), atol=1e-8)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_cdf_accepts_list_inputs(self):
        dist = WrappedCauchyDistribution(self.mu, self.gamma)
        xs = [0.5, 1.0, 2.0]

        npt.assert_allclose(dist.cdf(xs), dist.cdf(array(xs)), atol=1e-8)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_cdf_accepts_scalar_inputs(self):
        dist = WrappedCauchyDistribution(self.mu, self.gamma)

        npt.assert_allclose(dist.cdf(0.5), dist.cdf(array([0.5])), atol=1e-8)
        npt.assert_allclose(dist.cdf(array(0.5)), dist.cdf(array([0.5])), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
