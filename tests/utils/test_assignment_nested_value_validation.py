import unittest

import numpy as np
import pyrecest.backend
from pyrecest.utils import (
    min_cost_max_cardinality_assignment,
    murty_k_best_assignments,
)


def _nested_object_array(value, shape):
    result = np.empty(shape, dtype=object)
    result.flat[0] = np.asarray(value)
    return result


class AssignmentNestedValueValidationTest(unittest.TestCase):
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
    def test_nested_complex_cost_matrix_entries_are_rejected(self):
        matrix = _nested_object_array(1.0 + 2.0j, (1, 1))

        for solver_name, solver in self._solvers():
            with self.subTest(solver=solver_name):
                with self.assertRaisesRegex(ValueError, "real-valued"):
                    solver(matrix)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_nested_complex_non_assignment_costs_are_rejected(self):
        invalid_costs = _nested_object_array(1.0 + 2.0j, (1,))

        for name in ("row_non_assignment_costs", "col_non_assignment_costs"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "real-valued"):
                    murty_k_best_assignments(
                        np.array([[0.0]]),
                        k=1,
                        **{name: invalid_costs},
                    )

    @unittest.skipUnless(
        pyrecest.backend.__backend_name__ == "numpy",  # pylint: disable=no-member
        reason="Nested NumPy object arrays are only supported by the NumPy backend",
    )
    def test_nested_real_scalar_arrays_remain_supported(self):
        solutions = murty_k_best_assignments(
            _nested_object_array(0.25, (1, 1)),
            k=1,
            row_non_assignment_costs=_nested_object_array(2.0, (1,)),
            col_non_assignment_costs=_nested_object_array(0.0, (1,)),
        )

        self.assertEqual(len(solutions), 1)
        np.testing.assert_array_equal(solutions[0]["assignment"], np.array([0]))
        self.assertAlmostEqual(solutions[0]["cost"], 0.25)


if __name__ == "__main__":
    unittest.main()
