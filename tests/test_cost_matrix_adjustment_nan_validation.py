import unittest

import numpy as np
from pyrecest.utils.cost_matrix_adjustments import apply_cost_matrix_adjustment


class TestCostMatrixAdjustmentNanValidation(unittest.TestCase):
    def test_input_nan_costs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            apply_cost_matrix_adjustment(
                np.array([[np.nan]]),
                lambda matrix: matrix,
            )

    def test_adjustment_nan_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            apply_cost_matrix_adjustment(
                np.array([[1.0]]),
                lambda _matrix: np.array([[np.nan]]),
            )

    def test_positive_infinity_costs_remain_allowed(self):
        result = apply_cost_matrix_adjustment(
            np.array([[1.0, np.inf]]),
            lambda matrix: matrix,
        )

        self.assertTrue(np.isposinf(result.adjusted_cost_matrix[0, 1]))


if __name__ == "__main__":
    unittest.main()
