"""Regression tests for masked cost-matrix validation."""

import numpy as np
import pytest

from pyrecest.utils.cost_matrix_adjustments import (
    CostMatrixAdjustmentResult,
    additive_cost_matrix_adjustment,
    apply_cost_matrix_adjustment,
)


@pytest.mark.parametrize(
    "masked_matrix",
    [
        np.ma.array(
            [[1.0, 2.0], [3.0, 4.0]],
            mask=[[False, True], [False, False]],
        ),
        [[1.0, np.ma.masked], [3.0, 4.0]],
        np.array([[1.0, np.ma.masked], [3.0, 4.0]], dtype=object),
    ],
)
def test_public_entry_points_reject_masked_costs(masked_matrix):
    with pytest.raises(ValueError, match="real-valued numeric"):
        CostMatrixAdjustmentResult(masked_matrix)
    with pytest.raises(ValueError, match="real-valued numeric"):
        apply_cost_matrix_adjustment(masked_matrix, lambda matrix: matrix)
    with pytest.raises(ValueError, match="real-valued numeric"):
        additive_cost_matrix_adjustment(masked_matrix)


def test_adjustment_output_rejects_masked_costs():
    masked_output = np.ma.array([[1.0, 2.0]], mask=[[False, True]])

    with pytest.raises(ValueError, match="real-valued numeric"):
        apply_cost_matrix_adjustment(
            np.array([[0.0, 1.0]]),
            lambda _matrix: masked_output,
        )


def test_clear_mask_wrapper_remains_supported():
    matrix = np.ma.array([[1.0, np.inf], [3.0, 4.0]], mask=False)

    result = apply_cost_matrix_adjustment(matrix, lambda value: value)

    np.testing.assert_array_equal(
        result.adjusted_cost_matrix,
        np.array([[1.0, np.inf], [3.0, 4.0]]),
    )
