import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.utils.cost_matrix_adjustments import (
    CallableCostMatrixAdjustment,
    CostMatrixAdjustmentResult,
    apply_cost_matrix_adjustment,
)


def test_adjustment_accepts_none_diagnostics_as_empty():
    tuple_result = apply_cost_matrix_adjustment(
        np.array([[1.0]]),
        lambda matrix: (matrix + 1.0, None),
    )
    container_result = apply_cost_matrix_adjustment(
        np.array([[2.0]]),
        lambda matrix: CostMatrixAdjustmentResult(matrix + 2.0, None),
    )

    npt.assert_allclose(tuple_result.adjusted_cost_matrix, np.array([[2.0]]))
    npt.assert_allclose(container_result.adjusted_cost_matrix, np.array([[4.0]]))
    assert dict(tuple_result.diagnostics) == {}
    assert dict(container_result.diagnostics) == {}


def test_callable_adjustment_accepts_none_metadata():
    adjustment = CallableCostMatrixAdjustment(
        name="identity",
        function=lambda matrix, metadata: (matrix, metadata),
        metadata=None,
    )

    result = adjustment.apply(np.array([[1.0]]), metadata=None)

    npt.assert_allclose(result.adjusted_cost_matrix, np.array([[1.0]]))
    assert dict(result.diagnostics) == {}


def test_adjustment_rejects_invalid_diagnostics_mapping():
    with pytest.raises(ValueError, match="diagnostics must be a mapping or None"):
        CostMatrixAdjustmentResult(np.array([[1.0]]), object())
