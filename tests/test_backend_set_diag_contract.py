import pyrecest.backend as backend
import pytest


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_set_diag_accepts_array_like_matrix_inputs():
    result = backend.set_diag([[1, 2], [3, 4]], [9, 8])

    assert _to_python(result) == [[9, 2], [3, 8]]


def test_raw_pytorch_set_diag_accepts_array_like_matrix_inputs():
    pytest.importorskip("torch")

    import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel

    result = raw_pytorch.set_diag([[1, 2], [3, 4]], [9, 8])

    assert raw_pytorch.to_numpy(result).tolist() == [[9, 2], [3, 8]]
