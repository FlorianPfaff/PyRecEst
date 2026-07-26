import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.utils.cost_matrix_adjustments import CostMatrixAdjustmentResult


class TestCostMatrixAdjustmentResultValidation(unittest.TestCase):
    def test_result_takes_ownership_of_adjusted_matrix(self):
        matrix = np.array([[1.0, 2.0]], dtype=float)

        result = CostMatrixAdjustmentResult(matrix)
        matrix[:] = -7.0

        npt.assert_allclose(result.adjusted_cost_matrix, np.array([[1.0, 2.0]]))

    def test_result_rejects_invalid_direct_construction(self):
        invalid_matrices = (
            np.array([1.0, 2.0]),
            np.array([[np.nan]]),
            np.array([[-np.inf]]),
            np.array([[True]]),
            np.array([[1.0 + 0.0j]]),
        )

        for matrix in invalid_matrices:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    CostMatrixAdjustmentResult(matrix)


if __name__ == "__main__":
    unittest.main()
