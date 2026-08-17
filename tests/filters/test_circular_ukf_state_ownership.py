import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.circular_ukf import CircularUKF


class CircularUKFStateOwnershipTest(unittest.TestCase):
    @unittest.skipUnless(
        pyrecest.backend.__backend_name__ == "numpy",
        "mutable NumPy arrays required for aliasing regression",
    )
    def test_assignment_copies_state(self):
        filt = CircularUKF()
        assigned = GaussianDistribution(array([0.5]), array([[0.7]]))
        filt.filter_state = assigned

        self.assertIsNot(filt.filter_state, assigned)
        assigned.mu[0] = 1.5
        assigned.C[0, 0] = 2.0

        npt.assert_equal(filt.filter_state.mu, array([0.5]))
        npt.assert_equal(filt.filter_state.C, array([[0.7]]))


if __name__ == "__main__":
    unittest.main()
