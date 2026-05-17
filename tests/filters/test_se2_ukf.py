import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, concatenate, cos, eye, linalg, pi, sin, trace
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.se2_ukf import SE2UKF, _dual_quaternion_multiply


def _make_identity_state(scale=0.1):
    """Return a GaussianDistribution centred at the SE(2) identity."""
    mu = array([1.0, 0.0, 0.0, 0.0])
    C = eye(4) * scale
    return GaussianDistribution(mu, C)


def _make_noise(scale=0.01):
    """Return a small-noise GaussianDistribution at the SE(2) identity."""
    mu = array([1.0, 0.0, 0.0, 0.0])
    C = eye(4) * scale
    return GaussianDistribution(mu, C)


def _normalize_dq_mean(mean):
    """Normalize the rotation part of a 4-D dual-quaternion mean."""
    return concatenate([mean[0:2] / linalg.norm(mean[0:2]), mean[2:]])


class TestDualQuaternionMultiply(unittest.TestCase):
    def test_identity_left(self):
        """Multiplying by the identity from the left leaves dq unchanged."""
        identity = array([1.0, 0.0, 0.0, 0.0])
        dq = array([0.7071, 0.7071, 0.3, -0.1])
        result = _dual_quaternion_multiply(identity, dq)
        npt.assert_allclose(result, dq, atol=1e-10)

    def test_identity_right(self):
        """Multiplying by the identity from the right leaves dq unchanged."""
        identity = array([1.0, 0.0, 0.0, 0.0])
        dq = array([0.7071, 0.7071, 0.3, -0.1])
        result = _dual_quaternion_multiply(dq, identity)
        npt.assert_allclose(result, dq, atol=1e-10)

    def test_rotation_composition(self):
        """Composing two 90-degree rotations gives a 180-degree rotation."""
        theta = pi / 2
        dq_90 = array([cos(theta / 2), sin(theta / 2), 0.0, 0.0])
        result = _dual_quaternion_multiply(dq_90, dq_90)
        # cos(pi/2) == 0, sin(pi/2) == 1 (exact mathematical values)
        expected = array([0.0, 1.0, 0.0, 0.0])
        npt.assert_allclose(result, expected, atol=1e-6)

    def test_result_shape(self):
        dq1 = array([1.0, 0.0, 0.0, 0.0])
        dq2 = array([0.0, 1.0, 0.0, 0.0])
        result = _dual_quaternion_multiply(dq1, dq2)
        self.assertEqual(result.shape, (4,))


class TestSE2UKF(unittest.TestCase):
    def setUp(self):
        self.filter = SE2UKF()

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_initial_state_is_gaussian(self):
        self.assertIsInstance(self.filter.filter_state, GaussianDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_set_state(self):
        state = _make_identity_state()
        self.filter.filter_state = state
        fs = self.filter.filter_state
        self.assertIsInstance(fs, GaussianDistribution)
        npt.assert_allclose(fs.mu, state.mu, atol=1e-10)
        npt.assert_allclose(fs.C, state.C, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_set_state_unnormalised_raises(self):
        mu = array([1.0, 1.0, 0.0, 0.0])  # not unit-norm in first 2 entries
        C = eye(4) * 0.1
        with self.assertRaises(AssertionError):
            self.filter.filter_state = GaussianDistribution(mu, C)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_returns_gaussian(self):
        self.filter.filter_state = _make_identity_state()
        noise = _make_noise()
        self.filter.predict_identity(noise)
        self.assertIsInstance(self.filter.filter_state, GaussianDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_mean_normalised(self):
        """After prediction the rotation part of the mean must stay normalised."""
        self.filter.filter_state = _make_identity_state()
        self.filter.predict_identity(_make_noise())
        mu = self.filter.filter_state.mu
        npt.assert_allclose(linalg.norm(mu[0:2]), 1.0, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_mean_uses_system_noise_mean(self):
        """A non-identity process-noise mean must move the predicted mean."""
        self.filter.filter_state = _make_identity_state(scale=1e-12)

        theta = pi / 3.0
        motion_mu = array([cos(theta / 2.0), sin(theta / 2.0), 0.25, -0.1])
        motion = GaussianDistribution(motion_mu, eye(4) * 1e-12)

        self.filter.predict_identity(motion)

        expected = _normalize_dq_mean(
            _dual_quaternion_multiply(
                array([1.0, 0.0, 0.0, 0.0]),
                motion_mu,
            )
        )
        npt.assert_allclose(self.filter.filter_state.mu, expected, atol=1e-6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_uses_state_then_increment_order(self):
        """Prediction uses the documented right-increment order x [⊕] v."""
        state_theta = pi / 4.0
        increment_theta = pi / 6.0
        state_mu = array([cos(state_theta / 2.0), sin(state_theta / 2.0), 0.4, -0.2])
        increment_mu = array(
            [cos(increment_theta / 2.0), sin(increment_theta / 2.0), 0.1, 0.3]
        )
        self.filter.filter_state = GaussianDistribution(state_mu, eye(4) * 1e-12)
        increment = GaussianDistribution(increment_mu, eye(4) * 1e-12)

        self.filter.predict_identity(increment)

        expected = _normalize_dq_mean(_dual_quaternion_multiply(state_mu, increment_mu))
        reverse_order = _normalize_dq_mean(
            _dual_quaternion_multiply(increment_mu, state_mu)
        )
        self.assertGreater(float(linalg.norm(expected - reverse_order)), 1e-3)
        npt.assert_allclose(self.filter.filter_state.mu, expected, atol=1e-6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_covariance_shape(self):
        self.filter.filter_state = _make_identity_state()
        self.filter.predict_identity(_make_noise())
        C = self.filter.filter_state.C
        self.assertEqual(C.shape, (4, 4))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_covariance_symmetric(self):
        self.filter.filter_state = _make_identity_state()
        self.filter.predict_identity(_make_noise())
        C = self.filter.filter_state.C
        npt.assert_allclose(C, C.T, atol=1e-12)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_covariance_is_centered(self):
        """A near-deterministic prediction must not retain mu @ mu.T."""
        eps = 1e-12
        self.filter.filter_state = _make_identity_state(scale=eps)
        self.filter.predict_identity(_make_noise(scale=eps))

        C = self.filter.filter_state.C
        self.assertLess(float(trace(C)), 1e-8)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_covariance_increases(self):
        """Adding process noise should make the covariance (trace) larger."""
        self.filter.filter_state = _make_identity_state(scale=0.05)
        C_before = trace(self.filter.filter_state.C)
        self.filter.predict_identity(_make_noise(scale=0.05))
        C_after = trace(self.filter.filter_state.C)
        self.assertGreater(C_after, C_before)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_update_identity_returns_gaussian(self):
        self.filter.filter_state = _make_identity_state()
        z = array([1.0, 0.0, 0.0, 0.0])
        self.filter.update_identity(_make_noise(), z)
        self.assertIsInstance(self.filter.filter_state, GaussianDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_update_identity_mean_normalised(self):
        """After update the rotation part of the mean must stay normalised."""
        self.filter.filter_state = _make_identity_state()
        z = array([1.0, 0.0, 0.5, -0.3])
        z = concatenate([z[0:2] / linalg.norm(z[0:2]), z[2:]])
        self.filter.update_identity(_make_noise(), z)
        mu = self.filter.filter_state.mu
        npt.assert_allclose(linalg.norm(mu[0:2]), 1.0, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_update_identity_covariance_decreases(self):
        """An informative measurement should reduce uncertainty."""
        self.filter.filter_state = _make_identity_state(scale=0.5)
        C_before = trace(self.filter.filter_state.C)
        z = array([1.0, 0.0, 0.0, 0.0])
        self.filter.update_identity(_make_noise(scale=0.01), z)
        C_after = trace(self.filter.filter_state.C)
        self.assertLess(C_after, C_before)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_then_update_cycle(self):
        """A full predict+update cycle should leave the state as a GaussianDistribution
        with a normalised rotation part in the mean."""
        self.filter.filter_state = _make_identity_state()
        self.filter.predict_identity(_make_noise())
        z = array([1.0, 0.0, 0.1, -0.05])
        z = concatenate([z[0:2] / linalg.norm(z[0:2]), z[2:]])
        self.filter.update_identity(_make_noise(), z)
        self.assertIsInstance(self.filter.filter_state, GaussianDistribution)
        mu = self.filter.filter_state.mu
        npt.assert_allclose(linalg.norm(mu[0:2]), 1.0, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_get_point_estimate(self):
        self.filter.filter_state = _make_identity_state()
        est = self.filter.get_point_estimate()
        npt.assert_allclose(est, [1.0, 0.0, 0.0, 0.0], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
