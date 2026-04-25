import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.utils import murty_k_best_assignments


class MurtyAssignmentTest(unittest.TestCase):
    @staticmethod
    def _brute_force_solutions(
        cost_matrix,
        row_non_assignment_costs=None,
        col_non_assignment_costs=None,
    ):
        cost_matrix = np.asarray(cost_matrix, dtype=float)
        n_rows, n_cols = cost_matrix.shape
        if row_non_assignment_costs is None:
            row_non_assignment_costs = np.zeros(n_rows)
        if col_non_assignment_costs is None:
            col_non_assignment_costs = np.zeros(n_cols)

        row_non_assignment_costs = np.asarray(row_non_assignment_costs, dtype=float)
        col_non_assignment_costs = np.asarray(col_non_assignment_costs, dtype=float)

        solutions = []
        available_columns = range(n_cols)

        def recurse(row_index, used_columns, current_assignment, current_cost):
            if row_index == n_rows:
                current_cost += sum(
                    col_non_assignment_costs[col_index]
                    for col_index in available_columns
                    if col_index not in used_columns
                )
                assignment = np.asarray(current_assignment, dtype=int)
                solutions.append(
                    {
                        "assignment": assignment,
                        "unassigned_rows": np.where(assignment < 0)[0].astype(int),
                        "unassigned_cols": np.asarray(
                            [
                                col_index
                                for col_index in available_columns
                                if col_index not in used_columns
                            ],
                            dtype=int,
                        ),
                        "cost": float(current_cost),
                    }
                )
                return

            recurse(
                row_index + 1,
                used_columns,
                current_assignment + [-1],
                current_cost + row_non_assignment_costs[row_index],
            )
            for col_index in available_columns:
                if col_index in used_columns or not np.isfinite(
                    cost_matrix[row_index, col_index]
                ):
                    continue
                recurse(
                    row_index + 1,
                    used_columns | {col_index},
                    current_assignment + [col_index],
                    current_cost + cost_matrix[row_index, col_index],
                )

        recurse(0, set(), [], 0.0)

        unique_solutions = []
        seen_assignments = set()
        for solution in sorted(
            solutions,
            key=lambda current_solution: (
                current_solution["cost"],
                tuple(current_solution["assignment"]),
            ),
        ):
            assignment_key = tuple(int(value) for value in solution["assignment"])
            if assignment_key in seen_assignments:
                continue
            seen_assignments.add(assignment_key)
            unique_solutions.append(solution)

        return unique_solutions

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_matches_bruteforce_for_small_random_problems(self):
        rng = np.random.default_rng(0)

        for n_rows in range(4):
            for n_cols in range(4):
                for _ in range(5):
                    cost_matrix = rng.normal(size=(n_rows, n_cols))
                    if n_rows > 0 and n_cols > 0:
                        cost_matrix[rng.random(size=(n_rows, n_cols)) < 0.2] = np.inf
                    row_costs = rng.normal(size=n_rows)
                    col_costs = rng.normal(size=n_cols)

                    expected_solutions = self._brute_force_solutions(
                        cost_matrix,
                        row_costs,
                        col_costs,
                    )
                    actual_solutions = murty_k_best_assignments(
                        cost_matrix,
                        k=min(10, len(expected_solutions)),
                        row_non_assignment_costs=row_costs,
                        col_non_assignment_costs=col_costs,
                    )

                    self.assertEqual(
                        len(actual_solutions), min(10, len(expected_solutions))
                    )
                    for actual_solution, expected_solution in zip(
                        actual_solutions,
                        expected_solutions,
                    ):
                        npt.assert_array_equal(
                            actual_solution["assignment"],
                            expected_solution["assignment"],
                        )
                        npt.assert_array_equal(
                            actual_solution["unassigned_rows"],
                            expected_solution["unassigned_rows"],
                        )
                        npt.assert_array_equal(
                            actual_solution["unassigned_cols"],
                            expected_solution["unassigned_cols"],
                        )
                        self.assertAlmostEqual(
                            actual_solution["cost"],
                            expected_solution["cost"],
                        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_empty_column_case_returns_all_rows_unassigned(self):
        solutions = murty_k_best_assignments(
            np.empty((2, 0)),
            k=3,
            row_non_assignment_costs=np.array([1.0, 2.0]),
        )

        self.assertEqual(len(solutions), 1)
        npt.assert_array_equal(solutions[0]["assignment"], np.array([-1, -1]))
        npt.assert_array_equal(solutions[0]["unassigned_rows"], np.array([0, 1]))
        npt.assert_array_equal(solutions[0]["unassigned_cols"], np.array([], dtype=int))
        self.assertAlmostEqual(solutions[0]["cost"], 3.0)

    def test_non_positive_k_returns_empty_list(self):
        self.assertEqual(murty_k_best_assignments(np.eye(2), k=0), [])


if __name__ == "__main__":
    unittest.main()
