import unittest

import numpy as np
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


class GlobalNearestNeighborPairwiseCostValidationTest(unittest.TestCase):
    def test_rejects_non_real_pairwise_cost_matrix_dtypes(self):
        invalid_matrices = (
            np.array([[True]], dtype=bool),
            np.array([[1.0 + 1.0j]], dtype=complex),
            np.array([["1.0"]], dtype=str),
        )

        for invalid_matrix in invalid_matrices:
            with self.subTest(dtype=invalid_matrix.dtype):
                with self.assertRaisesRegex(
                    ValueError,
                    "pairwise_cost_matrix must contain real numeric costs",
                ):
                    GlobalNearestNeighbor._validate_pairwise_cost_matrix(
                        invalid_matrix,
                        1,
                        1,
                    )

    def test_rejects_nan_and_negative_infinite_pairwise_costs(self):
        invalid_matrices = (np.array([[np.nan]]), np.array([[-np.inf]]))

        for invalid_matrix in invalid_matrices:
            with self.subTest(invalid_matrix=invalid_matrix):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite values or positive infinity",
                ):
                    GlobalNearestNeighbor._validate_pairwise_cost_matrix(
                        invalid_matrix,
                        1,
                        1,
                    )

    def test_accepts_positive_infinity_as_pairwise_gate(self):
        pairwise_cost_matrix = GlobalNearestNeighbor._validate_pairwise_cost_matrix(
            np.array([[np.inf]]),
            1,
            1,
        )

        self.assertTrue(np.isposinf(np.asarray(pairwise_cost_matrix)[0, 0]))


if __name__ == "__main__":
    unittest.main()
