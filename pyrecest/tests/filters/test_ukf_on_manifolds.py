"""Tests for UKFOnManifolds."""

import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,redefined-builtin
import pyrecest.backend
from pyrecest.backend import all, array, diag, eye, linalg, pi, random, zeros
from pyrecest.filters.ukf_on_manifolds import UKFOnManifolds

# ---------------------------------------------------------------------------
# Helpers: Euclidean manifold (trivial phi / phi_inv, for numerical checks)
# ---------------------------------------------------------------------------


def _phi_euclidean(state, xi):
    return state + xi


def _phi_inv_euclidean(state_ref, state):
    return state - state_ref


# ---------------------------------------------------------------------------
# Helpers: SO(2) minimal example
# ---------------------------------------------------------------------------


def _phi_so2(state, xi):
    """Retraction on SO(2): state is a scalar angle, xi is a scalar tangent."""
    return state + xi[0]


def _phi_inv_so2(state_ref, state):
    """Inverse retraction on SO(2)."""
    diff = state - state_ref
    # wrap to [-pi, pi]
    diff = (diff + pi) % (2 * pi) - pi
    return array([diff])


class TestUKFOnManifoldsEuclidean(unittest.TestCase):
    """Verify the filter recovers Kalman-filter results on a Euclidean state space."""

    def _make_filter(self, d=1, alpha=1e-3):
        Q = eye(d) * 0.1
        R = eye(d) * 0.5
        state0 = zeros(d)
        P0 = eye(d)

        def f(s, omega, w, dt):  # pylint: disable=unused-argument
            return s + w

        def h(s):
            return s

        return UKFOnManifolds(
            f=f,
            h=h,
            phi=_phi_euclidean,
            phi_inv=_phi_inv_euclidean,
            Q=Q,
            R=R,
            alpha=alpha,
            state0=state0,
            P0=P0,
        )

    # ------------------------------------------------------------------
    def test_initialization_state(self):
        ukf = self._make_filter(d=2)
        state, P = ukf.filter_state
        npt.assert_array_equal(state, zeros(2))
        npt.assert_array_equal(P, eye(2))

    def test_get_point_estimate_initial(self):
        ukf = self._make_filter(d=2)
        npt.assert_array_equal(ukf.get_point_estimate(), zeros(2))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_increases_covariance(self):
        """After a predict step (no control), P should increase by Q."""
        ukf = self._make_filter(d=2)
        _, P_before = ukf.filter_state
        ukf.predict(omega=None, dt=1.0)
        _, P_after = ukf.filter_state
        # P should increase (P_after >= P_before elementwise for diagonal)
        self.assertTrue(all(diag(P_after) >= diag(P_before)))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_update_reduces_covariance(self):
        """After an update step, P should be smaller."""
        ukf = self._make_filter(d=2)
        ukf.predict(omega=None, dt=1.0)
        _, P_before_update = ukf.filter_state
        ukf.update(y=array([0.5, -0.5]))
        _, P_after_update = ukf.filter_state
        # Covariance should decrease after update
        self.assertTrue(all(diag(P_after_update) < diag(P_before_update)))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_update_moves_state_toward_measurement(self):
        """The state estimate should move toward the measurement after update."""
        ukf = self._make_filter(d=1)
        ukf.predict(omega=None, dt=1.0)
        y = array([2.0])
        state_before, _ = ukf.filter_state
        ukf.update(y=y)
        state_after, _ = ukf.filter_state
        # state should have moved in the direction of the measurement
        self.assertGreater(float(state_after[0]), float(state_before[0]))
        self.assertLess(float(state_after[0]), float(y[0]))

    def test_filter_state_setter_getter_roundtrip(self):
        ukf = self._make_filter(d=2)
        new_state = array([1.0, 2.0])
        new_P = diag(array([3.0, 4.0]))
        ukf.filter_state = (new_state, new_P)
        state_out, P_out = ukf.filter_state
        npt.assert_array_equal(state_out, new_state)
        npt.assert_array_equal(P_out, new_P)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_alpha_scalar_vs_array_equivalent(self):  # pylint: disable=too-many-locals
        """Passing a scalar alpha should give the same result as [alpha, alpha, alpha]."""
        Q = eye(2) * 0.1
        R = eye(2) * 0.5
        state0 = zeros(2)
        P0 = eye(2)
        measurement = array([1.0, -1.0])

        def f(s, omega, w, dt):  # pylint: disable=unused-argument
            return s + w

        def h(s):
            return s

        common = {
            "f": f,
            "h": h,
            "phi": _phi_euclidean,
            "phi_inv": _phi_inv_euclidean,
            "Q": Q,
            "R": R,
            "state0": state0,
            "P0": P0,
        }
        ukf_scalar = UKFOnManifolds(alpha=1e-3, **common)
        ukf_array = UKFOnManifolds(alpha=[1e-3, 1e-3, 1e-3], **common)

        for ukf in (ukf_scalar, ukf_array):
            ukf.predict(omega=None, dt=1.0)
            ukf.update(y=measurement)

        state_s, P_s = ukf_scalar.filter_state
        state_a, P_a = ukf_array.filter_state
        npt.assert_allclose(state_s, state_a)
        npt.assert_allclose(P_s, P_a)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_repeated_updates_converge(self):
        """Multiple measurements of the same value should converge the estimate."""
        ukf = self._make_filter(d=1, alpha=1e-3)
        y_true = array([5.0])
        for _ in range(20):
            ukf.predict(omega=None, dt=1.0)
            ukf.update(y=y_true + random.normal(loc=0.0, scale=0.1, size=(1,)))
        state, _ = ukf.filter_state
        # Should be close to 5
        npt.assert_allclose(state[0], 5.0, atol=1.0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_covariance_remains_symmetric_pd(self):
        """Covariance matrix must stay symmetric and positive-definite."""
        ukf = self._make_filter(d=3)
        for _ in range(10):
            ukf.predict(omega=None, dt=0.5)
            ukf.update(y=random.normal(loc=0.0, scale=1.0, size=(3,)))
        _, P = ukf.filter_state
        npt.assert_allclose(P, P.T, atol=1e-12)
        eigvals = linalg.eigvalsh(P)
        self.assertTrue(all(eigvals > 0))


class TestUKFOnManifoldsSO2(unittest.TestCase):
    """Test UKF-M on the circle SO(2) ~ angle."""

    def _make_filter(self):
        Q = array([[0.01]])  # 1×1 noise
        R = array([[0.1]])
        state0 = 0.0  # scalar angle
        P0 = array([[0.5]])

        def f(s, omega, w, dt):  # pylint: disable=unused-argument
            # constant-velocity on SO(2)
            return s + w[0]

        def h(s):
            return array([s])

        return UKFOnManifolds(
            f=f,
            h=h,
            phi=_phi_so2,
            phi_inv=_phi_inv_so2,
            Q=Q,
            R=R,
            alpha=1e-3,
            state0=state0,
            P0=P0,
        )

    def test_initialization(self):
        ukf = self._make_filter()
        state, P = ukf.filter_state
        self.assertAlmostEqual(float(state), 0.0)
        npt.assert_array_equal(P, array([[0.5]]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_update_moves_toward_measurement(self):
        ukf = self._make_filter()
        ukf.predict(omega=None, dt=1.0)
        y = array([0.3])
        state_before, _ = ukf.filter_state
        ukf.update(y=y)
        state_after, _ = ukf.filter_state
        self.assertGreater(float(state_after), float(state_before))
        self.assertLess(float(state_after), float(y[0]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_covariance_decreases_after_update(self):
        ukf = self._make_filter()
        ukf.predict(omega=None, dt=1.0)
        _, P_pred = ukf.filter_state
        ukf.update(y=array([0.1]))
        _, P_upd = ukf.filter_state
        self.assertLess(float(P_upd[0, 0]), float(P_pred[0, 0]))


class TestUKFOnManifoldsLinearEquivalence(unittest.TestCase):
    """
    For a linear Gaussian model on Euclidean space, UKF-M should give results
    very close to the exact Kalman filter.
    """

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_linear_1d_matches_kalman(self):
        """
        x_{k+1} = x_k + w_k,  w_k ~ N(0, Q)
        y_k     = x_k   + v_k, v_k ~ N(0, R)
        """
        Q_val = 0.1
        R_val = 0.5
        x0 = 0.0
        P0 = 1.0
        y = 2.0

        # --- UKF-M ---
        def f(s, omega, w, dt):  # pylint: disable=unused-argument
            return s + w

        def h(s):
            return s

        ukf = UKFOnManifolds(
            f=f,
            h=h,
            phi=lambda s, xi: s + xi[0],
            phi_inv=lambda s_ref, s: array([s - s_ref]),
            Q=array([[Q_val]]),
            R=array([[R_val]]),
            alpha=1e-3,
            state0=array([x0]),
            P0=array([[P0]]),
        )
        ukf.predict(omega=None, dt=1.0)
        ukf.update(y=array([y]))
        state_ukfm, P_ukfm = ukf.filter_state

        # --- Exact Kalman ---
        P_pred = P0 + Q_val
        K = P_pred / (P_pred + R_val)
        x_upd = x0 + K * (y - x0)
        P_upd = (1 - K) * P_pred

        npt.assert_allclose(float(state_ukfm[0]), x_upd, rtol=1e-5)
        npt.assert_allclose(float(P_ukfm[0, 0]), P_upd, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
