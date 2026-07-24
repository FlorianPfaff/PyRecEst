import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.utils.cost_matrix_adjustments import (
    additive_cost_matrix_adjustment,
    apply_cost_matrix_adjustment,
)


class TestAdditiveCostMatrixAdjustmentOwnership(unittest.TestCase):
    def test_caller_mutation_does_not_change_stored_penalty(self):
        penalty = np.array([[1.0, 2.0]])
        adjustment = additive_cost_matrix_adjustment(penalty)

        penalty[:] = 100.0

        result = apply_cost_matrix_adjustment(np.array([[10.0, 20.0]]), adjustment)
        npt.assert_allclose(result.adjusted_cost_matrix, np.array([[11.0, 22.0]]))


if __name__ == "__main__":
    unittest.main()
