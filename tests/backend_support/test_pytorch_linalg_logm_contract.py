import pytest

import pyrecest.backend as backend
from pyrecest.backend import linalg

pytorch_backend = pytest.importorskip("pyrecest._backend.pytorch")
pytorch_linalg = pytest.importorskip("pyrecest._backend.pytorch.linalg")


def _allclose_to_zero_matrix(values):
    expected = pytorch_backend.zeros((2, 2), dtype=values.dtype)
    return bool(pytorch_backend.allclose(values, expected))


def test_raw_pytorch_logm_accepts_array_like_integer_inputs():
    result = pytorch_linalg.logm([[1, 0], [0, 1]])

    assert result.shape == (2, 2)
    assert pytorch_backend.is_floating(result)
    assert _allclose_to_zero_matrix(result)


def test_public_pytorch_logm_accepts_array_like_integer_inputs_when_active():
    if getattr(backend, "__backend_name__", None) != "pytorch":
        pytest.skip("public PyTorch backend is not active")

    result = linalg.logm([[1, 0], [0, 1]])

    assert result.shape == (2, 2)
    assert backend.is_floating(result)
    assert bool(backend.allclose(result, backend.zeros((2, 2), dtype=result.dtype)))
