import numpy as np
import pytest

from pyrecest._backend.numpy import linalg


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.complex64, np.complex64),
        (np.int64, np.float64),
    ],
)
def test_sqrtm_empty_batches_preserve_shape_and_dtype(dtype, expected_dtype):
    matrices = np.empty((0, 3, 3), dtype=dtype)

    result = linalg.sqrtm(matrices)

    assert result.shape == matrices.shape
    assert result.dtype == expected_dtype
