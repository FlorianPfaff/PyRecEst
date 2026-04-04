import unittest

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

    def test_pdf(self):
        def pdftemp(x):
            return sum(
                self.lambda_ * exp(-self.lambda_ * (x + 2.0 * pi * k))
                for k in arange(-20, 21)
                if x + 2.0 * pi * k >= 0
            )

        for x in [0.0, 1.0, 2.0, 3.0, 4.0]:
            npt.assert_allclose(
                self.we.pdf(array(x)), pdftemp(array(x)), rtol=5e-7
            )

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
        npt.assert_allclose(
            self.we.entropy(), self.we.entropy_numerical(), rtol=5e-7
        )

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


if __name__ == "__main__":
    unittest.main()
