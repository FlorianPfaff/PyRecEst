import unittest
import warnings

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.utils import murty_k_best_assignments


class MurtyExtremeCostTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_returns_all_available_solutions_when_large_cost_overflows(self):
        cost_matrix = np.array([[1.0e308]])

        for requested_solutions in (2, 3):
            with self.subTest(requested_solutions=requested_solutions):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    solutions = murty_k_best_assignments(
                        cost_matrix,
                        k=requested_solutions,
                    )

                self.assertEqual(len(solutions), 2)
                npt.assert_array_equal(solutions[0]["assignment"], np.array([-1]))
                npt.assert_array_equal(solutions[0]["unassigned_rows"], np.array([0]))
                npt.assert_array_equal(solutions[0]["unassigned_cols"], np.array([0]))
                self.assertEqual(solutions[0]["cost"], 0.0)

                npt.assert_array_equal(solutions[1]["assignment"], np.array([0]))
                npt.assert_array_equal(
                    solutions[1]["unassigned_rows"], np.array([], dtype=int)
                )
                npt.assert_array_equal(
                    solutions[1]["unassigned_cols"], np.array([], dtype=int)
                )
                self.assertEqual(solutions[1]["cost"], 1.0e308)
