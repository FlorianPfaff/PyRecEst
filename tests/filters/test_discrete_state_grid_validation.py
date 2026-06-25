import unittest

import numpy as np
from pyrecest.filters.discrete_state import sparse_gaussian_transition_matrix


class TestDiscreteStateGridValidation(unittest.TestCase):
    def test_sparse_gaussian_transition_matrix_requires_finite_states(self):
        invalid_grids = (
            np.array([0.0, np.nan, 2.0]),
            np.array([[0.0], [np.inf], [2.0]]),
        )

        for grid in invalid_grids:
            with self.subTest(grid=grid):
                with self.assertRaisesRegex(ValueError, "state_vectors.*finite"):
                    sparse_gaussian_transition_matrix(grid, sigma=1.0)

    def test_sparse_gaussian_transition_matrix_rejects_non_real_numeric_states(self):
        invalid_grids = (
            np.array([True, False]),
            np.array([True, 1.0], dtype=object),
            np.array(["0.0", "1.0"]),
            np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        )

        for grid in invalid_grids:
            with self.subTest(grid=grid):
                with self.assertRaisesRegex(ValueError, "state_vectors.*real numeric"):
                    sparse_gaussian_transition_matrix(grid, sigma=1.0)


if __name__ == "__main__":
    unittest.main()
