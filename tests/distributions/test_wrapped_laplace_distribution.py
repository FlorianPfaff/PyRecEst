import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arange, array, exp, linspace, pi
from pyrecest.distributions.circle.wrapped_laplace_distribution import (
    WrappedLaplaceDistribution,
)


class WrappedLaplaceDistributionTest(unittest.TestCase):
    def setUp(self):
        self.lambda_ = array(2.0)
        self.kappa = array(1.3)
        self.wl = WrappedLaplaceDistribution(self.lambda_, self.kappa)

    def test_accepts_python_scalar_parameters(self):
        wl = WrappedLaplaceDistribution(2.0, 1.3)

        npt.assert_allclose(wl.pdf(array(1.0)), self.wl.pdf(array(1.0)), rtol=1e-6)

    def test_pdf(self):
        def laplace(x):
            return (
                self.lambda_
                / (1 / self.kappa + self.kappa)
                * exp(
                    -(
                        abs(x)
                        * self.lambda_
                        * (self.kappa if x >= 0 else 1 / self.kappa)
                    )
                )
            )

        def pdftemp(x):
            return sum(laplace(z) for z in x + 2.0 * pi * arange(-20, 21))

        for x in [0.0, 1.0, 2.0, 3.0, 4.0]:
            npt.assert_allclose(self.wl.pdf(array(x)), pdftemp(array(x)), rtol=1e-6)

    def test_pdf_accepts_scalar_and_list_inputs(self):
        npt.assert_allclose(self.wl.pdf(1.0), self.wl.pdf(array(1.0)), rtol=1e-6)
        npt.assert_allclose(
            self.wl.pdf([0.0, 1.0, 2.0]),
            self.wl.pdf(array([0.0, 1.0, 2.0])),
            rtol=1e-6,
        )

    def test_rejects_invalid_lambda(self):
        for lambda_ in (0.0, -0.5, float("inf"), array([1.0, 2.0])):
            with self.subTest(lambda_=lambda_):
                with self.assertRaisesRegex(ValueError, "lambda_"):
                    WrappedLaplaceDistribution(lambda_, self.kappa)

    def test_rejects_invalid_kappa(self):
        for kappa in (0.0, -0.5, float("inf"), array([1.0, 2.0])):
            with self.subTest(kappa=kappa):
                with self.assertRaisesRegex(ValueError, "kappa_"):
                    WrappedLaplaceDistribution(self.lambda_, kappa)

    def test_pdf_rejects_matrix_inputs(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            self.wl.pdf(array([[0.0, 1.0]]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_integral(self):
        npt.assert_allclose(self.wl.integrate(), 1.0, rtol=1e-10)
        npt.assert_allclose(self.wl.integrate_numerically(), 1.0, rtol=1e-10)
        npt.assert_allclose(
            self.wl.integrate(array([0.0, pi]))
            + self.wl.integrate(array([pi, 2.0 * pi])),
            1.0,
            rtol=1e-10,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_angular_moments(self):
        for i in range(1, 4):
            npt.assert_allclose(
                self.wl.trigonometric_moment(i),
                self.wl.trigonometric_moment_numerical(i),
                rtol=1e-10,
            )

    def test_periodicity(self):
        npt.assert_allclose(
            self.wl.pdf(linspace(-2.0 * pi, 0.0, 100)),
            self.wl.pdf(linspace(0.0, 2.0 * pi, 100)),
            rtol=5e-6,
        )


if __name__ == "__main__":
    unittest.main()
