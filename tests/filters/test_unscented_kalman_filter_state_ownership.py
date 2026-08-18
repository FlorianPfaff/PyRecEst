import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="UnscentedKalmanFilter is not supported on this backend",
)
class UnscentedKalmanFilterStateOwnershipTest(unittest.TestCase):
    def test_constructor_does_not_alias_caller_owned_state(self):
        initial_state = GaussianDistribution(array([1.0]), array([[2.0]]))
        filter_ = UnscentedKalmanFilter(initial_state)

        initial_state.mu[0] = 9.0
        initial_state.C[0, 0] = 7.0

        npt.assert_allclose(filter_.filter_state.mu, array([1.0]))
        npt.assert_allclose(filter_.filter_state.C, array([[2.0]]))

    def test_filter_state_assignment_does_not_alias_caller_owned_state(self):
        filter_ = UnscentedKalmanFilter(
            GaussianDistribution(array([0.0]), array([[1.0]]))
        )
        assigned_state = GaussianDistribution(array([3.0]), array([[4.0]]))

        filter_.filter_state = assigned_state
        assigned_state.mu[0] = -5.0
        assigned_state.C[0, 0] = 11.0

        npt.assert_allclose(filter_.filter_state.mu, array([3.0]))
        npt.assert_allclose(filter_.filter_state.C, array([[4.0]]))
