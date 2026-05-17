import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.circular_ukf import CircularUKF


class CircularUKFTest(unittest.TestCase):
    def setUp(self):
        self.filter = CircularUKF()
        self.g = GaussianDistribution(array([0.5]), array([[0.7]]))

    def test_initialization(self):
        self.filter.filter_state = self.g
        g1 = self.filter.filter_state
        self.assertIsInstance(g1, GaussianDistribution)
        npt.assert_equal(self.g.mu, g1.mu)
        npt.assert_equal(self.g.C, g1.C)

    def test_predict_identity(self):
        self.filter.filter_state = self.g
        self.filter.predict_identity(self.g)
        g_identity = self.filter.filter_state
        self.assertIsInstance(g_identity, GaussianDistribution)
        npt.assert_almost_equal(g_identity.mu, self.g.mu + self.g.mu)
        npt.assert_almost_equal(g_identity.C, self.g.C + self.g.C)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_predict_nonlinear_identity_function(self):
        self.filter.filter_state = self.g
        self.filter.predict_nonlinear(lambda x: x, self.g)
        g_nonlin = self.filter.filter_state
        self.assertIsInstance(g_nonlin, GaussianDistribution)
        npt.assert_almost_equal(g_nonlin.mu, self.g.mu + self.g.mu, decimal=10)
        npt.assert_almost_equal(g_nonlin.C, self.g.C + self.g.C, decimal=10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_predict_nonlinear_true_nonlinear(self):
        g4 = GaussianDistribution(array([0.0]), array([[0.7]]))
        self.filter.filter_state = g4
        self.filter.predict_nonlinear(lambda x: x**3, g4)
        g_nonlin = self.filter.filter_state
        # 0.0 and 2*pi are the same angle; compare via circular distance
        mu_diff = (float(g_nonlin.mu[0]) - float(g4.mu[0])) % (2.0 * float(pi))
        if mu_diff > float(pi):
            mu_diff -= 2.0 * float(pi)
        self.assertAlmostEqual(mu_diff, 0.0, places=10)
        self.assertGreater(float(g_nonlin.C[0, 0]), float(g4.C[0, 0]))

    def test_update_identity(self):
        self.filter.filter_state = self.g
        self.filter.update_identity(
            GaussianDistribution(array([0.0]), self.g.C), self.g.mu
        )
        g_identity = self.filter.filter_state
        self.assertIsInstance(g_identity, GaussianDistribution)
        npt.assert_almost_equal(g_identity.mu, self.g.mu)
        npt.assert_almost_equal(g_identity.C, self.g.C / 2.0)

    def test_update_identity_nonzero_noise_mean(self):
        self.filter.filter_state = self.g
        self.filter.update_identity(
            GaussianDistribution(array([2.0]), self.g.C), self.g.mu + array([2.0])
        )
        g_identity2 = self.filter.filter_state
        self.assertIsInstance(g_identity2, GaussianDistribution)
        npt.assert_almost_equal(g_identity2.mu, self.g.mu)
        npt.assert_almost_equal(g_identity2.C, self.g.C / 2.0)

    def test_update_identity_different_measurement(self):
        self.filter.filter_state = self.g
        z = 2.0 * pi - 1.0
        self.filter.update_identity(GaussianDistribution(array([0.0]), self.g.C), z)
        g_identity3 = self.filter.filter_state
        self.assertIsInstance(g_identity3, GaussianDistribution)
        self.assertGreater(float(g_identity3.mu[0]), z)
        self.assertLess(float(g_identity3.mu[0]), 2.0 * pi)
        self.assertGreater(float(self.g.C[0, 0]), float(g_identity3.C[0, 0]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_nonlinear_identity_function(self):
        g7 = GaussianDistribution(array([0.0]), array([[0.7]]))
        self.filter.filter_state = g7
        z = array([0.4])
        self.filter.update_nonlinear(lambda x: x, g7, z)
        g8 = self.filter.filter_state
        self.assertGreater(float(g8.mu[0]), float(g7.mu[0]))
        self.assertLess(float(g8.mu[0]), float(z[0]))
        self.assertGreater(float(g7.C[0, 0]), float(g8.C[0, 0]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_nonlinear_periodic_measurement(self):
        g9 = GaussianDistribution(array([0.1]), array([[0.7]]))
        self.filter.filter_state = g9
        z = array([2.0 * pi - 0.4])
        g7 = GaussianDistribution(array([0.0]), array([[0.7]]))
        self.filter.update_nonlinear(lambda x: x, g7, z, measurement_periodic=True)
        g10 = self.filter.filter_state
        self.assertGreater(float(g10.mu[0]), float(z[0]))
        self.assertLess(float(g10.mu[0]), 2.0 * pi)
        self.assertGreater(float(g7.C[0, 0]), float(g10.C[0, 0]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_nonlinear_periodic_measurement_uses_predicted_branch(self):
        prior = GaussianDistribution(array([0.1]), array([[0.7]]))
        meas_noise = GaussianDistribution(array([0.0]), array([[0.7]]))
        self.filter.filter_state = prior

        def shifted_periodic_measurement(x):
            return (x + 4.0) % (2.0 * float(pi))

        z = array([shifted_periodic_measurement(float(prior.mu[0]))])
        self.filter.update_nonlinear(
            shifted_periodic_measurement,
            meas_noise,
            z,
            measurement_periodic=True,
        )
        posterior = self.filter.filter_state
        diff = (float(posterior.mu[0] - prior.mu[0]) + float(pi)) % (
            2.0 * float(pi)
        ) - float(pi)
        self.assertAlmostEqual(diff, 0.0, places=8)

    def test_get_point_estimate(self):
        self.filter.filter_state = self.g
        estimate = self.filter.get_point_estimate()
        npt.assert_equal(estimate, self.g.mu)


if __name__ == "__main__":
    unittest.main()
