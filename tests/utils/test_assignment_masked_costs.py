import unittest

import numpy as np
import pyrecest.backend
from pyrecest.utils import (
    min_cost_max_cardinality_assignment,
    murty_k_best_assignments,
)


class AssignmentMaskedCostValidationTest(unittest.TestCase):
    @staticmethod
    def _solvers():
        return (
            ("murty", lambda matrix: murty_k_best_assignments(matrix, k=1)),
            ("max_cardinality", min_cost_max_cardinality_assignment),
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_masked_cost_matrix_entries_are_rejected(self):
        invalid_matrices = (
            np.ma.array([[0.0, 1.0]], mask=[[False, True]]),
            np.array([[0.0, np.ma.masked]], dtype=object),
        )

        for solver_name, solver in self._solvers():
            for matrix in invalid_matrices:
                with self.subTest(solver=solver_name, matrix_type=type(matrix).__name__):
                    with self.assertRaisesRegex(
                        ValueError, "cost_matrix must not contain masked values"
                    ):
                        solver(matrix)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_masked_non_assignment_costs_are_rejected(self):
        matrix = np.array([[1.0, 2.0]])
        invalid_costs = (
            {"row_non_assignment_costs": np.ma.array([0.5], mask=[True])},
            {
                "col_non_assignment_costs": np.array(
                    [0.5, np.ma.masked], dtype=object
                )
            },
        )

        for kwargs in invalid_costs:
            name = next(iter(kwargs))
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError, f"{name} must not contain masked values"
                ):
                    murty_k_best_assignments(matrix, k=1, **kwargs)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_masked_assignment_count_is_rejected(self):
        for k in (np.ma.masked, np.ma.array(1, mask=True)):
            with self.subTest(k=repr(k)):
                with self.assertRaisesRegex(ValueError, "k must be an integer"):
                    murty_k_best_assignments(np.array([[1.0]]), k=k)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_fully_unmasked_masked_arrays_remain_supported(self):
        matrix = np.ma.array([[1.0, 2.0]], mask=False)
        for solver_name, solver in self._solvers():
            with self.subTest(solver=solver_name):
                result = solver(matrix)
                if solver_name == "murty":
                    self.assertEqual(len(result), 1)
                else:
                    np.testing.assert_array_equal(result["assignment"], np.array([0]))

        solutions = murty_k_best_assignments(
            matrix,
            k=np.ma.array(1, mask=False),
            row_non_assignment_costs=np.ma.array([3.0], mask=False),
            col_non_assignment_costs=np.ma.array([0.0, 0.0], mask=False),
        )
        self.assertEqual(len(solutions), 1)
        np.testing.assert_array_equal(solutions[0]["assignment"], np.array([0]))


if __name__ == "__main__":
    unittest.main()
