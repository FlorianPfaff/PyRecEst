import numpy as np
from pyrecest._backend.numpy import linalg


def test_sqrtm_promotes_integer_inputs_to_float_results():
    matrix = np.array([[2, 0], [0, 8]], dtype=np.int64)

    result = linalg.sqrtm(matrix)

    assert result.dtype.kind == "f"
    np.testing.assert_allclose(result, np.diag(np.sqrt([2.0, 8.0])))


def test_fractional_matrix_power_promotes_integer_inputs_to_float_results():
    matrix = np.array([[2, 0], [0, 8]], dtype=np.int64)

    result = linalg.fractional_matrix_power(matrix, 0.5)

    assert result.dtype.kind == "f"
    np.testing.assert_allclose(result, np.diag(np.sqrt([2.0, 8.0])))


def test_logm_promotes_nonsymmetric_integer_inputs_to_float_results():
    matrix = np.array([[2, 1], [0, 2]], dtype=np.int64)

    result = linalg.logm(matrix)

    assert result.dtype.kind == "f"
    expected = np.array([[np.log(2.0), 0.5], [0.0, np.log(2.0)]])
    np.testing.assert_allclose(result, expected)
