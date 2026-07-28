import pyrecest.backend as backend
import pytest
from pyrecest.backend import array


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_vec_to_diag_accepts_array_like_inputs():
    result = backend.vec_to_diag([1, 2, 3])

    assert result.shape == (3, 3)
    assert _to_python(result) == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]


def test_triangular_vector_helpers_accept_array_like_inputs():
    if backend.__backend_name__ == "pytorch":
        pytest.skip("PyTorch triangular array-like contract is covered separately")

    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    assert _to_python(backend.tril_to_vec(matrix)) == [1, 4, 5, 7, 8, 9]
    assert _to_python(backend.triu_to_vec(matrix)) == [1, 2, 3, 5, 6, 9]


def test_squeeze_non_singleton_axis_matches_backend_contract():
    values = array([[[1], [2]]])

    if backend.__backend_name__ in {"numpy", "autograd"}:
        with pytest.raises(ValueError, match="size not equal to one"):
            backend.squeeze(values, axis=1)
        return

    result = backend.squeeze(values, axis=1)

    assert result.shape == values.shape
    assert _to_python(result) == [[[1], [2]]]
