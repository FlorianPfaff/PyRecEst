import math
import unittest

import matplotlib
import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arange, array, column_stack, cos, exp, pi
from pyrecest.distributions.hypertorus.toroidal_von_mises_cosine_distribution import (
    ToroidalVonMisesCosineDistribution,
)
from scipy.special import iv
from tests.distributions.test_toroidal_von_mises_sine_distribution import (
    ToroidalBivarVMTestMixin,
)

matplotlib.pyplot.close("all")
matplotlib.use("Agg")


def _cosine_norm_const_reference(kappa, kappa3, n_terms=120):
    kappa0 = float(kappa[0])
    kappa1 = float(kappa[1])
    kappa3 = float(kappa3)
    terms = [iv(0, kappa0) * iv(0, kappa1) * iv(0, -kappa3)]
    terms.extend(
        2.0 * iv(order, kappa0) * iv(order, kappa1) * iv(order, -kappa3)
        for order in range(1, n_terms + 1)
    )
    return 4.0 * math.pi**2 * math.fsum(terms)


def _cosine_moment_reference(mu, kappa, kappa3, n_terms=120):
    kappa0 = float(kappa[0])
    kappa1 = float(kappa[1])
    kappa3 = float(kappa3)
    terms = range(1, n_terms + 1)

    def s(order):
        return iv(order, kappa0) * iv(order, kappa1) * iv(order, -kappa3)

    def s1(order):
        return (
            (iv(order + 1, kappa0) + iv(order - 1, kappa0))
            * iv(order, kappa1)
            * iv(order, -kappa3)
        )

    def s2(order):
        return (
            iv(order, kappa0)
            * (iv(order + 1, kappa1) + iv(order - 1, kappa1))
            * iv(order, -kappa3)
        )

    s_sum = math.fsum([s(0), *(2.0 * s(order) for order in terms)])
    s1_sum = math.fsum([s1(0) / 2.0, *(s1(order) for order in terms)])
    s2_sum = math.fsum([s2(0) / 2.0, *(s2(order) for order in terms)])
    m1 = s1_sum / s_sum * np.exp(1j * float(mu[0]))
    m2 = s2_sum / s_sum * np.exp(1j * float(mu[1]))
    return np.array([m1, m2])


class ToroidalVMCosineDistributionTest(ToroidalBivarVMTestMixin, unittest.TestCase):
    def setUp(self):
        self.mu = array([1.0, 2.0])
        self.kappa = array([0.7, 1.4])
        self.kappa3 = array(0.5)
        self.tvm = ToroidalVonMisesCosineDistribution(self.mu, self.kappa, self.kappa3)

    def test_instance(self):
        self.assertIsInstance(self.tvm, ToroidalVonMisesCosineDistribution)

    def test_mu_kappa_kappa3(self):
        npt.assert_allclose(self.tvm.mu, self.mu)
        npt.assert_allclose(self.tvm.kappa, self.kappa)
        self.assertEqual(self.tvm.kappa3, self.kappa3)

    def _unnormalized_pdf(self, xs):
        return exp(
            self.kappa[0] * cos(xs[..., 0] - self.mu[0])
            + self.kappa[1] * cos(xs[..., 1] - self.mu[1])
            - self.kappa3 * cos(xs[..., 0] - self.mu[0] - xs[..., 1] + self.mu[1])
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_trigonometric_moment_analytical(self):
        m_analytical = self.tvm.trigonometric_moment(1)
        m_numerical = self.tvm.trigonometric_moment_numerical(1)
        npt.assert_allclose(m_analytical, m_numerical, rtol=1e-8)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ != "numpy",
        reason="Regression test uses NumPy/scipy scalar semantics",
    )
    def test_norm_const_uses_adaptive_series_when_needed(self):
        kappa = array([20.0, 20.0])
        kappa3 = array(10.0)
        tvm = ToroidalVonMisesCosineDistribution(array([0.5, 1.0]), kappa, kappa3)
        npt.assert_allclose(
            tvm.norm_const,
            _cosine_norm_const_reference(kappa, kappa3),
            rtol=1e-10,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ != "numpy",
        reason="Regression test uses NumPy/scipy scalar semantics",
    )
    def test_trigonometric_moment_uses_adaptive_series_when_needed(self):
        mu = array([0.5, 1.0])
        kappa = array([20.0, 20.0])
        kappa3 = array(10.0)
        tvm = ToroidalVonMisesCosineDistribution(mu, kappa, kappa3)
        npt.assert_allclose(
            tvm.trigonometric_moment(1),
            _cosine_moment_reference(mu, kappa, kappa3),
            rtol=1e-10,
        )

    def test_shift(self):
        shift_by = array([4.0, 2.0])
        tvm2 = self.tvm.shift(shift_by)
        self.assertIsInstance(tvm2, ToroidalVonMisesCosineDistribution)
        x_test = column_stack((arange(0.0, 2.0 * pi, 0.3), arange(0.0, 2.0 * pi, 0.3)))
        npt.assert_allclose(
            tvm2.pdf(x_test),
            self.tvm.pdf(x_test - shift_by),
            atol=1e-10,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
