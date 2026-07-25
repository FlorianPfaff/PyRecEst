import numpy as np
import pytest
from pyrecest._backend.numpy import linalg


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        pytest.param(np.float32, np.float32, id="float32"),
        pytest.param(np.complex64, np.complex64, id="complex64"),
        pytest.param(np.int64, np.float64, id="integer-promoted"),
    ],
)
def test_sylvester_preserves_broadcast_empty_batches(dtype, expected_dtype):
    a = np.empty((2, 0, 1, 2, 2), dtype=dtype)
    b = np.empty((1, 0, 4, 3, 3), dtype=dtype)
    q = np.empty((2, 1, 4, 2, 3), dtype=dtype)

    result = linalg.solve_sylvester(a, b, q)

    assert result.shape == (2, 0, 4, 2, 3)
    assert result.dtype == np.dtype(expected_dtype)


def test_sylvester_rejects_empty_batches_with_invalid_core_shapes():
    a = np.empty((0, 2, 3))
    b = np.empty((0, 4, 4))
    q = np.empty((0, 2, 4))

    with pytest.raises(ValueError):
        linalg.solve_sylvester(a, b, q)
