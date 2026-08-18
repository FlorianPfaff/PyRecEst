"""Regression tests for UKFOnManifolds state ownership."""

import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye, zeros
from pyrecest.filters.ukf_on_manifolds import UKFOnManifolds


def _make_filter(state0):
    def f(state, omega, noise, dt):  # pylint: disable=unused-argument
        return state

    def h(state):  # pylint: disable=unused-argument
        return zeros(1)

    def phi(state, xi):  # pylint: disable=unused-argument
        return state

    def phi_inv(state_ref, state):  # pylint: disable=unused-argument
        return zeros(1)

    return UKFOnManifolds(
        f=f,
        h=h,
        phi=phi,
        phi_inv=phi_inv,
        Q=eye(1),
        R=eye(1),
        alpha=1e-3,
        state0=state0,
        P0=eye(1),
    )


class TestUKFOnManifoldsStateOwnership(unittest.TestCase):
    def test_constructor_does_not_retain_caller_owned_state(self):
        state0 = {"position": [1.0]}
        ukf = _make_filter(state0)

        state0["position"][0] = 9.0

        state, _ = ukf.filter_state
        self.assertEqual(state["position"][0], 1.0)
        self.assertIsNot(state, state0)

    def test_filter_state_assignment_copies_mutable_state(self):
        ukf = _make_filter({"position": [0.0]})
        assigned_state = {"position": [2.0]}

        ukf.filter_state = (assigned_state, eye(1))
        assigned_state["position"][0] = 8.0

        state, _ = ukf.filter_state
        self.assertEqual(state["position"][0], 2.0)
        self.assertIsNot(state, assigned_state)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="JAX arrays are immutable, so caller-side covariance mutation cannot be tested.",
    )
    def test_filter_state_assignment_copies_covariance(self):
        ukf = _make_filter({"position": [0.0]})
        assigned_covariance = array([[2.0]])

        ukf.filter_state = ({"position": [3.0]}, assigned_covariance)
        assigned_covariance[0, 0] = 7.0

        _, covariance = ukf.filter_state
        npt.assert_allclose(covariance, array([[2.0]]))


if __name__ == "__main__":
    unittest.main()
