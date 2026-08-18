import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter


class UnscentedKalmanFilterStateOwnershipTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_initial_and_assigned_gaussian_states_are_copied(self):
        initial_state = GaussianDistribution(array([1.0]), array([[2.0]]))
        ukf = UnscentedKalmanFilter(initial_state)

        initial_state.mu[0] = 10.0
        initial_state.C[0, 0] = 20.0

        npt.assert_allclose(ukf.filter_state.mu, array([1.0]))
        npt.assert_allclose(ukf.filter_state.C, array([[2.0]]))

        assigned_state = GaussianDistribution(array([3.0]), array([[4.0]]))
        ukf.filter_state = assigned_state

        assigned_state.mu[0] = 30.0
        assigned_state.C[0, 0] = 40.0

        npt.assert_allclose(ukf.filter_state.mu, array([3.0]))
        npt.assert_allclose(ukf.filter_state.C, array([[4.0]]))


if __name__ == "__main__":
    unittest.main()
