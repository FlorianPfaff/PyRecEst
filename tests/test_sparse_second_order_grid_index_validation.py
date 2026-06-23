import math

import numpy as np
import pytest
from pyrecest.filters.sparse_second_order_grid import sparse_second_order_grid_evidence


def _log_likelihood():
    return np.zeros((3, 2), dtype=float)


def _valid_initial_pairs(scaled):
    del scaled
    return np.array([0]), np.array([0]), np.array([1.0]), [1]


def _valid_transition_row(prev, curr, transition_index):
    del prev, curr, transition_index
    return np.array([0]), np.array([1.0])


@pytest.mark.parametrize(
    ("prev_indices", "curr_indices"),
    [
        (np.array([0.5]), np.array([0])),
        (np.array([0]), np.array([True])),
        (np.array(["0"], dtype=object), np.array([0])),
    ],
)
def test_sparse_second_order_grid_rejects_non_integral_initial_indices(
    prev_indices, curr_indices
):
    def init(scaled):
        del scaled
        return prev_indices, curr_indices, np.array([1.0]), [1]

    with pytest.raises(ValueError, match="initial pair indices must be integer-valued"):
        sparse_second_order_grid_evidence(
            _log_likelihood(), init, _valid_transition_row
        )


@pytest.mark.parametrize(
    "dst_indices",
    [
        np.array([0.5]),
        np.array([True]),
        np.array(["0"], dtype=object),
    ],
)
def test_sparse_second_order_grid_rejects_non_integral_transition_destinations(
    dst_indices,
):
    def row(prev, curr, transition_index):
        del prev, curr, transition_index
        return dst_indices, np.array([1.0])

    with pytest.raises(
        ValueError, match="transition row destination indices must be integer-valued"
    ):
        sparse_second_order_grid_evidence(_log_likelihood(), _valid_initial_pairs, row)


def test_sparse_second_order_grid_preserves_integer_valued_float_indices():
    def init(scaled):
        del scaled
        return np.array([0.0]), np.array([1.0]), np.array([1.0]), [1]

    def row(prev, curr, transition_index):
        del prev, curr, transition_index
        return np.array([1.0]), np.array([1.0])

    result = sparse_second_order_grid_evidence(_log_likelihood(), init, row)

    assert math.isfinite(result.log_marginal_likelihood)
    np.testing.assert_allclose(np.exp(result.terminal_log_probabilities).sum(), 1.0)
