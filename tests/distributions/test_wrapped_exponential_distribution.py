import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arange, arctan, array, exp, linspace, pi
from pyrecest.distributions.circle.wrapped_exponential_distribution import (
    WrappedExponentialDistribution,
)


class WrappedExponentialDistributionTest(unittest.TestCase):
    def setUp(self):
        self.lambda_ = array(2.0)
        self.we = WrappedExponentialDistribution(self.lambda_)

    def test_accepts_python_scalar_parameter(self):
        we = WrappedExponentialDistribution(2.0)

        npt.assert_allclose(we.pdf(array(1.0)), self.we.pdf(array(1.0)), rtol=5e-7)

    def test_pdf(self):
        def pdftemp(x):
            return sum(
                self.lambda_ * exp(-self.lambda_ * (x + 2.0 * pi * k))
                for k in arange(-20, 21)
                if x + 2.0 * pi * k >= 0
            )

        for x in [0.0, 1.0, 2.0, 3.0, 4.0]:
            npt.assert_allclose(self.we.pdf(array(x)), pdftemp(array(x)), rtol=5e-7)

    def test_pdf_accepts_scalar_and_list_inputs(self):
        npt.assert_allclose(self.we.pdf(1.0), self.we.pdf(array(1.0)), rtol=5e-7)
        npt.assert_allclose(
            self.we.pdf([0.0, 1.0, 2.0]),
            self.we.pdf(array([0.0, 1.0, 2.0])),
            rtol=5e-7,
        )

    def test_rejects_invalid_lambda(self):
        for lambda_ in (0.0, -0.5, float("inf"), array([1.0, 2.0])):
            with self.subTest(lambda_=lambda_):
                with self.assertRaisesRegex(ValueError, "lambda_"):
                    WrappedExponentialDistribution(lambda_)

    def test_pdf_rejects_matrix_inputs(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            self.we.pdf(array([[0.0, 1.0]]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_integral(self):
        npt.assert_allclose(self.we.integrate(), 1.0, rtol=5e-7)
        npt.assert_allclose(self.we.integrate_numerically(), 1.0, rtol=5e-7)
        npt.assert_allclose(
            self.we.integrate(array([0.0, pi]))
            + self.we.integrate(array([pi, 2.0 * pi])),
            1.0,
            rtol=5e-7,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_angular_moments(self):
        for i in range(1, 4):
            npt.assert_allclose(
                self.we.trigonometric_moment(i),
                self.we.trigonometric_moment_numerical(i),
                rtol=5e-7,
            )

    def test_circular_mean(self):
        npt.assert_allclose(
            self.we.mean_direction(), float(arctan(1.0 / self.lambda_)), rtol=5e-7
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_entropy(self):
        npt.assert_allclose(self.we.entropy(), self.we.entropy_numerical(), rtol=5e-7)

    def test_periodicity(self):
        npt.assert_allclose(
            self.we.pdf(linspace(-2.0 * pi, 0.0, 100)),
            self.we.pdf(linspace(0.0, 2.0 * pi, 100)),
            rtol=5e-6,
        )

    def test_sample(self):
        n = 100
        s = self.we.sample(n)
        self.assertEqual(s.shape, (n,))
        self.assertTrue((s >= 0).all())
        self.assertTrue((s < 2.0 * pi).all())

    def test_sample_accepts_integer_like_count(self):
        samples = self.we.sample(np.int64(4))

        self.assertEqual(samples.shape, (4,))

    def test_sample_rejects_invalid_count(self):
        for n in (0, -1, 1.5, True):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.we.sample(n)


if __name__ == "__main__":
    unittest.main()
