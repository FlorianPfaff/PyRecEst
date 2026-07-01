import math
import unittest
from unittest.mock import patch

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, random
from pyrecest.distributions import WrappedNormalDistribution
from pyrecest.filters.wrapped_normal_filter import WrappedNormalFilter


def _scalar_only_likelihood(z, x):
    return math.exp(math.cos(float(z) - float(x)))


class WrappedNormalFilterTest(unittest.TestCase):
    def setUp(self):
        self.wn_filter = WrappedNormalFilter()
        self.wn = WrappedNormalDistribution(array(1.3), array(0.8))
        self.meas_noise = WrappedNormalDistribution(array(0.0), array(0.9))
        self.wn_filter.filter_state = self.wn

    def test_initialization(self):
        wn1 = self.wn_filter.filter_state
        self.assertIsInstance(wn1, WrappedNormalDistribution)
        self.assertEqual(self.wn.mu, wn1.mu)
        self.assertEqual(self.wn.sigma, wn1.sigma)

    def test_predict_identity(self):
        self.wn_filter.predict_identity(
            WrappedNormalDistribution(array(0.0), self.wn.sigma)
        )
        wn_identity = self.wn_filter.filter_state
        self.assertIsInstance(wn_identity, WrappedNormalDistribution)
        self.assertEqual(self.wn.mu, wn_identity.mu)
        self.assertLess(self.wn.sigma, wn_identity.sigma)

    def test_update(self):
        # update identity
        self.wn_filter.update_identity(self.meas_noise, self.wn.mu)
        wn_identity = self.wn_filter.filter_state
        self.assertIsInstance(wn_identity, WrappedNormalDistribution)
        npt.assert_almost_equal(self.wn.mu, wn_identity.mu)
        self.assertGreater(self.wn.sigma, wn_identity.sigma)

        # reset filter state for the next test within this function
        self.wn_filter.filter_state = self.wn

        # update identity with different measurement
        self.wn_filter.update_identity(self.meas_noise, self.wn.mu + 0.1)
        wn_identity2 = self.wn_filter.filter_state
        self.assertIsInstance(wn_identity2, WrappedNormalDistribution)
        self.assertLess(self.wn.mu, wn_identity2.mu)
        self.assertGreater(self.wn.sigma, wn_identity2.sigma)

    def test_update_nonlinear_particle_accepts_scalar_likelihood_callback(self):
        random.seed(0)
        self.wn_filter.filter_state = WrappedNormalDistribution(array(0.0), array(0.5))

        self.wn_filter.update_nonlinear_particle(_scalar_only_likelihood, 0.1)

        self.assertIsInstance(self.wn_filter.filter_state, WrappedNormalDistribution)
        self.assertTrue(math.isfinite(float(self.wn_filter.filter_state.sigma)))

    def test_update_nonlinear_progressive_accepts_scalar_likelihood_callback(self):
        self.wn_filter.filter_state = WrappedNormalDistribution(array(0.0), array(0.5))

        self.wn_filter.update_nonlinear_progressive(_scalar_only_likelihood, 0.1)

        self.assertIsInstance(self.wn_filter.filter_state, WrappedNormalDistribution)
        self.assertTrue(math.isfinite(float(self.wn_filter.filter_state.sigma)))

    def test_update_nonlinear_progressive_skips_constant_nonzero_likelihood(self):
        class FakeDiracDistribution:
            def __init__(self):
                self.d = array([0.0, 1.0])
                self.w = array([1.0, 1.0])
                self.reweigh_calls = 0

            def reweigh(self, weight_fun):
                self.reweigh_calls += 1
                npt.assert_allclose(weight_fun(self.d), array([2.0, 2.0]), rtol=1e-12)
                return self

            @staticmethod
            def to_wn():
                return WrappedNormalDistribution(array(0.0), array(1.0))

        fake_dirac = FakeDiracDistribution()

        def constant_likelihood(_z, _x):
            return 2.0

        with patch.object(
            WrappedNormalDistribution, "to_dirac5", return_value=fake_dirac
        ):
            self.wn_filter.update_nonlinear_progressive(constant_likelihood, z=0.0)

        self.assertEqual(fake_dirac.reweigh_calls, 0)

    def test_update_nonlinear_progressive_rejects_invalid_tau(self):
        self.wn_filter.filter_state = WrappedNormalDistribution(array(0.0), array(0.5))
        invalid_taus = (
            0.0,
            -0.1,
            float("nan"),
            float("inf"),
            True,
            "0.02",
            array([0.02]),
        )

        for tau in invalid_taus:
            with self.subTest(tau=tau), self.assertRaisesRegex(
                ValueError, "tau must be a positive finite scalar"
            ):
                self.wn_filter.update_nonlinear_progressive(
                    _scalar_only_likelihood, 0.1, tau=tau
                )

    def test_update_nonlinear_progressive_uses_adaptive_lambda(self):
        class FakeDiracDistribution:
            def __init__(self):
                self.d = array([0.0, 1.0])
                self.w = array([1.0, 1.0])
                self.reweigh_calls = 0

            def reweigh(self, weight_fun):
                self.reweigh_calls += 1

                # With equal weights, tau=0.02, and likelihood ratio 0.5,
                # the adaptive exponent is greater than the remaining lambda,
                # so the method should use the full remaining exponent 1.0.
                npt.assert_allclose(weight_fun(0.0), 0.5, rtol=1e-12)
                npt.assert_allclose(weight_fun(1.0), 1.0, rtol=1e-12)
                return self

            @staticmethod
            def to_wn():
                return WrappedNormalDistribution(array(0.0), array(1.0))

        fake_dirac = FakeDiracDistribution()

        def likelihood(_z, x):
            return 1.0 if x > 0.0 else 0.5

        with patch.object(
            WrappedNormalDistribution, "to_dirac5", return_value=fake_dirac
        ):
            self.wn_filter.update_nonlinear_progressive(likelihood, z=0.0)

        self.assertEqual(fake_dirac.reweigh_calls, 1)


if __name__ == "__main__":
    unittest.main()
