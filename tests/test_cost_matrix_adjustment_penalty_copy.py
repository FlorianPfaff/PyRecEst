import numpy as np
import numpy.testing as npt

from pyrecest.utils.cost_matrix_adjustments import (
    additive_cost_matrix_adjustment,
    apply_cost_matrix_adjustment,
)


def test_additive_adjustment_owns_penalty_matrix():
    penalty = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_penalty = penalty.copy()
    adjustment = additive_cost_matrix_adjustment(penalty)

    penalty[:] = -100.0

    result = apply_cost_matrix_adjustment(np.zeros((2, 2)), adjustment)
    npt.assert_array_equal(result.adjusted_cost_matrix, expected_penalty)
