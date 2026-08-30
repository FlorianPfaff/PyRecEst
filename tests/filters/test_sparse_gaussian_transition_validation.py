import numpy as np
import pytest

from pyrecest.filters.discrete_state import sparse_gaussian_transition_matrix


@pytest.mark.parametrize(
    ("state_vectors", "message"),
    [
        (np.array(1.0), "state_vectors must have shape"),
        (np.empty((2, 1, 1)), "state_vectors must have shape"),
        (np.empty((0,)), "state_vectors must contain at least one state"),
        (
            np.empty((2, 0)),
            "state_vectors must contain at least one coordinate per state",
        ),
    ],
)
def test_sparse_gaussian_transition_matrix_rejects_invalid_state_shapes(
    state_vectors, message
):
    with pytest.raises(ValueError, match=message):
        sparse_gaussian_transition_matrix(state_vectors, sigma=1.0)
