import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import array, exp, linspace, log, mean, pi, sum
from pyrecest.distributions.circle.piecewise_constant_distribution import (
    PiecewiseConstantDistribution,
)
from pyrecest.distributions.circle.wrapped_normal_distribution import (
    WrappedNormalDistribution,
)


class PiecewiseConstantDistributionTest(unittest.TestCase):
    def setUp(self):
        self.w = array([1, 2, 3, 4, 5, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        self.normal = 1.0 / (2.0 * pi * mean(self.w))
        self.dist = PiecewiseConstantDistribution(self.w)

    def test_pdf(self):
        npt.assert_allclose(
            self.dist.pdf(array([0.0])), array([1 * self.normal]), rtol=1e-10
        )
        npt.assert_allclose(
            self.dist.pdf(array([4.2])), array([5 * self.normal]), rtol=1e-10
        )
        npt.assert_allclose(
            self.dist.pdf(array([10.9])), array([4 * self.normal]), rtol=1e-10
        )

    def test_integral_normalized(self):
        """Verify the distribution integrates to 1 via the exact sum."""
        n = len(self.dist.w)
        npt.assert_allclose(sum(self.dist.w) * (2.0 * pi / n), 1.0, rtol=5e-7)

    def test_integral_partial(self):
        """Verify partial integrals sum to 1 using a fine grid."""
        M = 1_000_000
        xs = linspace(0, 2.0 * pi, M, endpoint=False)
        dx = 2.0 * pi / M
        pdf_vals = self.dist.pdf(xs)
        first_half = sum(pdf_vals[xs < pi]) * dx
        second_half = sum(pdf_vals[xs >= pi]) * dx
        npt.assert_allclose(first_half + second_half, 1.0, rtol=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_trigonometric_moment(self):
        """Verify analytical trigonometric moments using a fine-grid reference."""
        M = 1_000_000
        xs = linspace(0, 2.0 * pi, M, endpoint=False)
        pdf_vals = self.dist.pdf(xs)
        dx = 2.0 * pi / M
        for n_moment in [1, 2, 3]:
            expected = sum(pdf_vals * exp(1j * n_moment * xs)) * dx
            with self.subTest(n=n_moment):
                npt.assert_allclose(
                    self.dist.trigonometric_moment(n_moment), expected, rtol=1e-4
                )

    def test_interval_borders(self):
        self.assertAlmostEqual(
            PiecewiseConstantDistribution.left_border(1, 2), 0.0 * 2.0 * pi
        )
        self.assertAlmostEqual(
            PiecewiseConstantDistribution.interval_center(1, 2),
            1.0 / 4.0 * 2.0 * pi,
        )
        self.assertAlmostEqual(
            PiecewiseConstantDistribution.right_border(1, 2),
            1.0 / 2.0 * 2.0 * pi,
        )
        self.assertAlmostEqual(
            PiecewiseConstantDistribution.left_border(2, 2),
            1.0 / 2.0 * 2.0 * pi,
        )
        self.assertAlmostEqual(
            PiecewiseConstantDistribution.interval_center(2, 2),
            3.0 / 4.0 * 2.0 * pi,
        )
        self.assertAlmostEqual(
            PiecewiseConstantDistribution.right_border(2, 2), 1.0 * 2.0 * pi
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_calculate_parameters_numerically(self):
        """More samples should yield better moment matching."""
        wn = WrappedNormalDistribution(array(2.0), array(1.3))
        w1 = PiecewiseConstantDistribution.calculate_parameters_numerically(wn.pdf, 40)
        w2 = PiecewiseConstantDistribution.calculate_parameters_numerically(wn.pdf, 45)
        w3 = PiecewiseConstantDistribution.calculate_parameters_numerically(wn.pdf, 50)
        p1 = PiecewiseConstantDistribution(w1)
        p2 = PiecewiseConstantDistribution(w2)
        p3 = PiecewiseConstantDistribution(w3)
        delta1 = abs(wn.trigonometric_moment(1) - p1.trigonometric_moment(1))
        delta2 = abs(wn.trigonometric_moment(1) - p2.trigonometric_moment(1))
        delta3 = abs(wn.trigonometric_moment(1) - p3.trigonometric_moment(1))
        self.assertLessEqual(delta2, delta1)
        self.assertLessEqual(delta3, delta2)

    def test_entropy(self):
        """Verify analytical entropy against the direct formula."""
        w = self.dist.w
        n = len(w)
        expected = -2.0 * pi / n * sum(w * log(w))
        npt.assert_allclose(self.dist.entropy(), expected, rtol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_sample(self):
        samples = self.dist.sample(array(100))
        self.assertEqual(len(samples), 100)
        self.assertTrue(all(samples >= 0.0))
        self.assertTrue(all(samples < 2.0 * pi))


if __name__ == "__main__":
    unittest.main()
