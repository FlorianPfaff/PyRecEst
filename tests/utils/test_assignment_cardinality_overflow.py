import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, to_numpy
from pyrecest.utils import min_cost_max_cardinality_assignment


class MaxCardinalityAssignmentOverflowTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_extreme_finite_costs_do_not_overflow_priority_penalty(self):
        probe = np.asarray(to_numpy(array([0.0], dtype=float)))
        maximum = np.finfo(probe.dtype).max
        cost_matrix = array(
            [[0.0, maximum], [maximum, 0.0]],
            dtype=float,
        )

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            solution = min_cost_max_cardinality_assignment(cost_matrix)

        npt.assert_array_equal(
            np.asarray(to_numpy(solution["assignment"])),
            np.array([0, 1]),
        )
        npt.assert_array_equal(
            np.asarray(to_numpy(solution["unassigned_rows"])),
            np.array([], dtype=int),
        )
        npt.assert_array_equal(
            np.asarray(to_numpy(solution["unassigned_cols"])),
            np.array([], dtype=int),
        )
        self.assertEqual(solution["cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
