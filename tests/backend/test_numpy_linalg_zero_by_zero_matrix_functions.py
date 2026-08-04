import numpy as np
import pytest
from pyrecest._backend.numpy import linalg


@pytest.mark.parametrize(
    ("matrix_function", "args"),
    [
        pytest.param(linalg.logm, (), id="logm"),
        pytest.param(linalg.sqrtm, (), id="sqrtm"),
        pytest.param(
            linalg.fractional_matrix_power,
            (0.5,),
            id="fractional_matrix_power",
        ),
    ],
)
@pytest.mark.parametrize("shape", [(0, 0), (3, 0, 0), (2, 1, 0, 0)])
@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        pytest.param(np.float32, np.float32, id="float32"),
        pytest.param(np.complex64, np.complex64, id="complex64"),
        pytest.param(np.int64, np.float64, id="integer-promoted"),
    ],
)
def test_zero_by_zero_matrix_functions_preserve_shape_and_dtype(
    matrix_function, args, shape, dtype, expected_dtype
):
    matrices = np.empty(shape, dtype=dtype)

    result = matrix_function(matrices, *args)

    assert result.shape == shape
    assert result.dtype == np.dtype(expected_dtype)
