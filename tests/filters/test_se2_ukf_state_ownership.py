"""Regression tests for SE2UKF state ownership."""

import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.se2_ukf import SE2UKF


class TestSE2UKFStateOwnership(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="SE2UKF state assignment is not supported on JAX in the existing suite.",
    )
    def test_filter_state_assignment_does_not_retain_input_distribution(self):
        ukf = SE2UKF()
        assigned = GaussianDistribution(
            array([1.0, 0.0, 0.25, -0.5]),
            eye(4) * 0.2,
        )

        ukf.filter_state = assigned
        assigned.mu = array([0.0, 1.0, 9.0, 8.0])
        assigned.C = eye(4) * 3.0

        self.assertIsNot(ukf.filter_state, assigned)
        npt.assert_allclose(
            ukf.filter_state.mu,
            array([1.0, 0.0, 0.25, -0.5]),
        )
        npt.assert_allclose(ukf.filter_state.C, eye(4) * 0.2)


if __name__ == "__main__":
    unittest.main()
