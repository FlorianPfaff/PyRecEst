from __future__ import annotations

import numpy as np
import pytest
from pyrecest.utils.cost_matrix_adjustments import (
    additive_cost_matrix_adjustment,
    apply_cost_matrix_adjustment,
)

TEMPORAL_MATRICES = (
    np.array([[np.timedelta64(2, "ns")]]),
    np.array([[np.datetime64("1970-01-01T00:00:00.000000002")]]),
    np.array([[np.timedelta64(2, "ns")]], dtype=object),
    np.array([[np.datetime64("1970-01-01T00:00:00.000000002")]], dtype=object),
    [[np.timedelta64(2, "ns")]],
    [[np.datetime64("1970-01-01T00:00:00.000000002")]],
)


@pytest.mark.parametrize("matrix", TEMPORAL_MATRICES)
def test_cost_matrix_adjustment_rejects_temporal_input_matrices(matrix) -> None:
    with pytest.raises(ValueError, match="real-valued numeric"):
        apply_cost_matrix_adjustment(matrix, lambda costs: costs)


@pytest.mark.parametrize("matrix", TEMPORAL_MATRICES)
def test_additive_cost_matrix_adjustment_rejects_temporal_penalty_matrices(
    matrix,
) -> None:
    with pytest.raises(ValueError, match="real-valued numeric"):
        additive_cost_matrix_adjustment(matrix)


@pytest.mark.parametrize("matrix", TEMPORAL_MATRICES[:4])
def test_cost_matrix_adjustment_rejects_temporal_adjustment_outputs(matrix) -> None:
    with pytest.raises(ValueError, match="real-valued numeric"):
        apply_cost_matrix_adjustment(np.array([[0.0]]), lambda _costs: matrix)
