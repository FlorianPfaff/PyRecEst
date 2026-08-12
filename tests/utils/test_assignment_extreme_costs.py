import unittest
import warnings

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.utils import (
    min_cost_max_cardinality_assignment,
    murty_k_best_assignments,
)


class MurtyExtremeCostTest(unittest.TestCase):
    @staticmethod
    def _canceling_diagonal_cost_matrix():
        cost_matrix = np.full((4, 4), np.inf)
        np.fill_diagonal(
            cost_matrix,
            np.array([1.0e308, 1.0e308, -1.0e308, -1.0e308]),
        )
        return cost_matrix

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

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_murty_preserves_canceling_extreme_total(self):
        cost_matrix = self._canceling_diagonal_cost_matrix()
        non_assignment_costs = np.full(4, 1.0e308)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            solutions = murty_k_best_assignments(
                cost_matrix,
                k=1,
                row_non_assignment_costs=non_assignment_costs,
                col_non_assignment_costs=non_assignment_costs,
            )

        self.assertEqual(len(solutions), 1)
        npt.assert_array_equal(solutions[0]["assignment"], np.arange(4))
        self.assertEqual(solutions[0]["cost"], 0.0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_max_cardinality_preserves_canceling_extreme_total(self):
        cost_matrix = self._canceling_diagonal_cost_matrix()

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            solution = min_cost_max_cardinality_assignment(cost_matrix)

        npt.assert_array_equal(solution["assignment"], np.arange(4))
        self.assertEqual(solution["cost"], 0.0)
