import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    array,
    diag,
    exp,
    eye,
    ones,
    ones_like,
    pi,
    random,
    sum,
    zeros_like,
)
from pyrecest.distributions.cart_prod.hypercylindrical_dirac_distribution import (
    HypercylindricalDiracDistribution,
)
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)
from scipy.stats import wishart

from .test_abstract_dirac_distribution import TestAbstractDiracDistribution


class TestHypercylindricalDiracDistribution(TestAbstractDiracDistribution):
    def setUp(self):
        self.d = array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [2.0, 4.0, 0.0, 0.5, 1.0, 1.0],
                [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            ]
        ).T
        self.w = array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        self.w = self.w / sum(self.w)
        self.pwd = HypercylindricalDiracDistribution(1, self.d, self.w)

    def test_mean_and_marginalization(self):
        mean = self.pwd.hybrid_moment()
        wd = self.pwd.marginalize_linear()
        npt.assert_allclose(mean[0], wd.trigonometric_moment(1).real, rtol=1e-10)
        npt.assert_allclose(mean[1], wd.trigonometric_moment(1).imag, rtol=1e-10)
        npt.assert_allclose(mean[2], sum(self.w * self.d[:, 1]), rtol=1e-10)
        npt.assert_allclose(mean[3], sum(self.w * self.d[:, 2]), rtol=1e-7)

    def test_covariance(self):
        clin = self.pwd.linear_covariance()
        assert clin.shape == (self.pwd.lin_dim, self.pwd.lin_dim)

    def test_apply_function_identity(self):
        same = self.pwd.apply_function(lambda x: x)
        npt.assert_array_equal(self.pwd.d, same.d)
        npt.assert_array_equal(self.pwd.w, same.w)
        assert self.pwd.lin_dim == same.lin_dim
        assert self.pwd.bound_dim == same.bound_dim

    def test_apply_function_shift(self):
        shift_offset = array([1.4, -0.3, 1.0])

        def shift(x, shift_by=shift_offset):
            return x + shift_by

        shifted = self.pwd.apply_function(shift)
        npt.assert_allclose(
            shifted.marginalize_linear().trigonometric_moment(1),
            self.pwd.marginalize_linear().trigonometric_moment(1)
            * exp(1j * shift_offset[0]),
            atol=5e-7,
        )

    def test_reweigh(self):
        # Define functions for testing
        def f1(x):
            return sum(x, axis=-1) == 3

        def f2(x):
            return 2 * ones(x.shape[0])

        def f3(x):
            return x[:, 0]

        pwd_rew1 = self.pwd.reweigh(f1)
        pwd_rew2 = self.pwd.reweigh(f2)
        pwd_rew3 = self.pwd.reweigh(f3)

        assert isinstance(pwd_rew1, HypercylindricalDiracDistribution)
        npt.assert_array_equal(pwd_rew1.d, self.pwd.d)
        npt.assert_array_equal(pwd_rew1.w, f1(self.pwd.d))

        assert isinstance(pwd_rew2, HypercylindricalDiracDistribution)
        npt.assert_array_equal(pwd_rew2.d, self.pwd.d)
        npt.assert_array_equal(pwd_rew2.w, self.pwd.w)

        assert isinstance(pwd_rew3, HypercylindricalDiracDistribution)
        npt.assert_array_equal(pwd_rew3.d, self.pwd.d)
        w_new = self.pwd.d[:, 0] * self.pwd.w
        npt.assert_array_equal(pwd_rew3.w, w_new / sum(w_new))

    def test_sampling(self):
        random.seed(0)
        n = 10
        s = self.pwd.sample(n)
        assert s.shape == (n, 3)
        s = s[:, 0]
        self.assertTrue(all(s >= zeros_like(s)))
        self.assertTrue(all(s < 2 * pi * ones_like(s)))

    def test_from_distribution(self):
        import numpy as _np

        random_gen = _np.random.default_rng(0)  # Could fail randomly otherwise
        df = 4
        scale = eye(4)
        # Call array(...) to be compatibel with all backends
        C = array(wishart.rvs(df, scale, random_state=random_gen))
        hwn = PartiallyWrappedNormalDistribution(array([1.0, 2.0, 3.0, 4.0]), C, 2)
        hddist = HypercylindricalDiracDistribution.from_distribution(hwn, 200000)
        npt.assert_allclose(hddist.hybrid_mean(), hwn.hybrid_mean(), atol=0.2)

    # jscpd:ignore-start
    @parameterized.expand(
        [
            (
                "1D Plot",
                PartiallyWrappedNormalDistribution(
                    array([1.0]), array([[1.0]]), bound_dim=1  # 1D mean
                ),  # 1D covariance
                1,  # Dimension
            ),
            (
                "2D Plot",
                PartiallyWrappedNormalDistribution(
                    array([1.0, 2.0]),  # 2D mean
                    array([[2.0, -0.3], [-0.3, 1.0]]),
                    bound_dim=1,
                ),  # 2D covariance
                2,  # Dimension
            ),
            (
                "3D Plot",
                PartiallyWrappedNormalDistribution(
                    array([1.0, 2.0, 3.0]),  # 3D mean
                    diag(array([2.0, 1.0, 0.5])),
                    bound_dim=1,
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
        self._test_plot_helper(
            name, dist, dim, HypercylindricalDiracDistribution, bound_dim=1
        )

    # jscpd:ignore-end
