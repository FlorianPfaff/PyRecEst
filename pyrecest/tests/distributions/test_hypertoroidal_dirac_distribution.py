import copy
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import array, diag, exp, mod, pi, random, sum, zeros_like
from pyrecest.distributions import (
    AbstractHypertoroidalDistribution,
    HypertoroidalDiracDistribution,
    HypertoroidalWrappedNormalDistribution,
    ToroidalDiracDistribution,
)

from .test_abstract_dirac_distribution import TestAbstractDiracDistribution


class TestHypertoroidalDiracDistribution(TestAbstractDiracDistribution):
    def setUp(self):
        self.d = array(
            [[0.5, 2, 0.5], [3, 2, 0.2], [4, 5, 5.8], [6, 3, 4.3], [6, 0, 1.2]]
        )
        self.w = array([0.1, 0.1, 0.1, 0.1, 0.6])
        self.twd = HypertoroidalDiracDistribution(self.d, self.w)

    def test_init(self):
        self.assertIsInstance(self.twd, HypertoroidalDiracDistribution)
        npt.assert_array_almost_equal(self.twd.d, self.d)
        npt.assert_array_almost_equal(self.twd.w, self.w)

    def test_trigonometric_moment(self):
        m = self.twd.trigonometric_moment(1)
        m1 = self.twd.marginalize_to_1D(0).trigonometric_moment(1)
        m2 = self.twd.marginalize_to_1D(1).trigonometric_moment(1)
        npt.assert_almost_equal(m[0], m1, decimal=10)
        npt.assert_almost_equal(m[1], m2, decimal=10)
        npt.assert_almost_equal(m[0], sum(self.w * exp(1j * self.d[:, 0])), decimal=10)
        npt.assert_almost_equal(m[1], sum(self.w * exp(1j * self.d[:, 1])), decimal=10)

    def test_sample(self):
        n_samples = 5
        s = self.twd.sample(n_samples)
        self.assertEqual(s.shape, (n_samples, self.d.shape[-1]))
        npt.assert_array_almost_equal(s, mod(s, 2.0 * pi))

    def test_marginalize_to_1D(self):
        for i in range(self.d.shape[-1]):
            wd = self.twd.marginalize_to_1D(i)
            npt.assert_array_almost_equal(self.twd.w, wd.w)
            npt.assert_array_almost_equal(wd.d, self.twd.d[:, i])

    def test_apply_function(self):
        same = self.twd.apply_function(lambda x: x)
        npt.assert_array_almost_equal(
            same.trigonometric_moment(1), self.twd.trigonometric_moment(1)
        )
        shift_offset = array([1.4, -0.3, pi])
        shifted = self.twd.apply_function(lambda x: x + shift_offset)
        npt.assert_allclose(
            shifted.trigonometric_moment(1)[0],
            sum(self.w * exp(1j * (self.d[:, 0] + shift_offset[0]))),
            rtol=1e-6,
        )
        npt.assert_allclose(
            shifted.trigonometric_moment(1)[1],
            sum(self.w * exp(1j * (self.d[:, 1] + shift_offset[1]))),
            rtol=1e-10,
        )

    def test_shift(self):
        d = array(
            [
                [4, -2, 0.01],
                [3, 2, 0],
                [2.5, 4, -2.4],
                [8, 1 / 3, 2.2],
                [2.1, 7 / 99, 0.2],
            ]
        )

        w = array([0.3, 0.3, 0.3, 0.05, 0.05])
        twd = HypertoroidalDiracDistribution(d, w)
        s = array([1.0, -3.0, 6.0])
        twd_shifted = twd.shift(s)
        self.assertIsInstance(twd_shifted, HypertoroidalDiracDistribution)
        npt.assert_array_almost_equal(twd.w, twd_shifted.w)
        npt.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(twd.d, twd_shifted.d - s),
            zeros_like(twd.d),
            atol=1e-6,
        )

    @staticmethod
    def get_pseudorandom_hypertoroidal_wd(dim=2):
        random.seed(0)
        n = 20
        d = 2.0 * pi * random.uniform(size=(n, dim))
        w = random.uniform(size=n)
        w = w / sum(w)
        hwd = HypertoroidalDiracDistribution(d, w)
        return hwd

    def test_to_toroidal_wd(self):
        hwd = TestHypertoroidalDiracDistribution.get_pseudorandom_hypertoroidal_wd(2)
        twd1 = ToroidalDiracDistribution(copy.copy(hwd.d), copy.copy(hwd.w))
        twd2 = hwd.to_toroidal_wd()
        self.assertIsInstance(twd2, ToroidalDiracDistribution)
        npt.assert_array_almost_equal(twd1.d, twd2.d, decimal=10)
        npt.assert_array_almost_equal(twd1.w, twd2.w, decimal=10)

    def test_marginalization(self):
        hwd = TestHypertoroidalDiracDistribution.get_pseudorandom_hypertoroidal_wd(2)
        wd1 = hwd.marginalize_to_1D(0)
        wd2 = hwd.marginalize_out(1)
        npt.assert_array_almost_equal(wd1.d, wd2.d)
        npt.assert_array_almost_equal(wd1.w, wd2.w)

    # jscpd:ignore-start
    @parameterized.expand(
        [
            (
                "1D Plot",
                HypertoroidalWrappedNormalDistribution(
                    array([1.0]), array([[1.0]])  # 1D mean
                ),  # 1D covariance
                1,  # Dimension
            ),
            (
                "2D Plot",
                HypertoroidalWrappedNormalDistribution(
                    array([1.0, 2.0]), array([[2.0, -0.3], [-0.3, 1.0]])  # 2D mean
                ),  # 2D covariance
                2,  # Dimension
            ),
            (
                "3D Plot",
                HypertoroidalWrappedNormalDistribution(
                    array([1.0, 2.0, 3.0]), diag(array([2.0, 1.0, 0.5]))  # 3D mean
                ),  # 3D covariance (diagonal matrix)
                3,  # Dimension
            ),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_plot(self, name, dist, dim):
        self._test_plot_helper(name, dist, dim, HypertoroidalDiracDistribution)

    # jscpd:ignore-end


if __name__ == "__main__":
    unittest.main()
