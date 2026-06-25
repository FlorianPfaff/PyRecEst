import numpy as np
import pytest
from pyrecest.filters.sparse_second_order_grid import sparse_second_order_grid_evidence


def test_sparse_second_order_grid_rejects_boolean_initial_pair_weights():
    def init(scaled):
        del scaled
        return np.array([0]), np.array([0]), np.array([True]), [1]

    def row(prev, curr, transition_index):
        del prev, curr, transition_index
        return np.array([0]), np.array([1.0])

    with pytest.raises(ValueError, match="initial pair weights"):
        sparse_second_order_grid_evidence(np.zeros((3, 2)), init, row)


def test_sparse_second_order_grid_rejects_boolean_transition_row_weights():
    def init(scaled):
        del scaled
        return np.array([0]), np.array([0]), np.array([1.0]), [1]

    def row(prev, curr, transition_index):
        del prev, curr, transition_index
        return np.array([0]), np.array([True])

    with pytest.raises(ValueError, match="transition row weights"):
        sparse_second_order_grid_evidence(np.zeros((3, 2)), init, row)


def test_sparse_second_order_grid_rejects_text_initial_pair_weights():
    numeric_text = "".join(("1", ".", "0"))

    def init(scaled):
        del scaled
        return np.array([0]), np.array([0]), np.array([numeric_text], dtype=object), [1]

    def row(prev, curr, transition_index):
        del prev, curr, transition_index
        return np.array([0]), np.array([1.0])

    with pytest.raises(ValueError, match="initial pair weights"):
        sparse_second_order_grid_evidence(np.zeros((3, 2)), init, row)


def test_sparse_second_order_grid_rejects_text_transition_row_weights():
    numeric_text = "".join(("1", ".", "0"))

    def init(scaled):
        del scaled
        return np.array([0]), np.array([0]), np.array([1.0]), [1]

    def row(prev, curr, transition_index):
        del prev, curr, transition_index
        return np.array([0]), np.array([numeric_text], dtype=object)

    with pytest.raises(ValueError, match="transition row weights"):
        sparse_second_order_grid_evidence(np.zeros((3, 2)), init, row)
