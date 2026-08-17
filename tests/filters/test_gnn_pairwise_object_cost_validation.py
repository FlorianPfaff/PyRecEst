import unittest

import numpy as np
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


class GlobalNearestNeighborPairwiseObjectCostValidationTest(unittest.TestCase):
    @staticmethod
    def _object_matrix(value):
        matrix = np.empty((1, 1), dtype=object)
        matrix[0, 0] = value
        return matrix

    def test_rejects_non_real_values_hidden_in_object_arrays(self):
        invalid_values = (
            None,
            True,
            np.bool_(True),
            "1.0",
            b"1.0",
            bytearray(b"1.0"),
            np.str_("1.0"),
            np.bytes_(b"1.0"),
            1.0 + 0.0j,
            np.complex128(1.0 + 0.0j),
            np.datetime64("2026-07-27"),
            np.timedelta64(3, "ns"),
        )

        for invalid_value in invalid_values:
            with self.subTest(value=invalid_value, type=type(invalid_value).__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "pairwise_cost_matrix must contain real numeric costs",
                ):
                    GlobalNearestNeighbor._validate_pairwise_cost_matrix(
                        self._object_matrix(invalid_value),
                        1,
                        1,
                    )

    def test_rejects_invalid_values_hidden_by_mixed_type_coercion(self):
        invalid_matrices = (
            [[1.0, True]],
            [[1.0, "2.0"]],
            [[1.0, np.datetime64("2026-07-27")]],
            [[1.0, np.timedelta64(3, "ns")]],
        )

        for invalid_matrix in invalid_matrices:
            with self.subTest(matrix=invalid_matrix):
                with self.assertRaisesRegex(
                    ValueError,
                    "pairwise_cost_matrix must contain real numeric costs",
                ):
                    GlobalNearestNeighbor._validate_pairwise_cost_matrix(
                        invalid_matrix,
                        1,
                        2,
                    )

    def test_rejects_native_temporal_dtypes(self):
        invalid_matrices = (
            np.array([["2026-07-27"]], dtype="datetime64[D]"),
            np.array([[3]], dtype="timedelta64[ns]"),
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

    def test_accepts_real_numeric_values(self):
        pairwise_cost_matrix = GlobalNearestNeighbor._validate_pairwise_cost_matrix(
            [[1, 2.5]],
            1,
            2,
        )

        np.testing.assert_allclose(pairwise_cost_matrix, [[1.0, 2.5]])


if __name__ == "__main__":
    unittest.main()
