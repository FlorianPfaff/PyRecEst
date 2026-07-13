import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.utils.point_set_registration import solve_gated_assignment


class TestNegativeGatedAssignmentCosts(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_default_infinite_gate_matches_negative_cost(self):
        assignment = solve_gated_assignment(array([[-10.0]]))

        npt.assert_array_equal(assignment, array([0]))
