"""Regression tests for PyTorch linalg.logm input normalization."""

import pyrecest.backend as backend
import pytest

pytorch_backend = pytest.importorskip("pyrecest._backend.pytorch")
pytorch_linalg = pytest.importorskip("pyrecest._backend.pytorch.linalg")
torch = pytest.importorskip("torch")


def test_raw_pytorch_logm_accepts_array_like_integer_inputs():
    result = pytorch_linalg.logm([[1, 0], [0, 1]])

    assert result.dtype.is_floating_point
    assert torch.allclose(result, torch.zeros_like(result))


def test_raw_pytorch_logm_preserves_float32_complex_precision():
    matrix = torch.diag(torch.tensor([-1.0, 1.0], dtype=torch.float32))

    result = pytorch_linalg.logm(matrix)

    expected = torch.zeros((2, 2), dtype=torch.complex64)
    expected[0, 0] = 1j * torch.pi
    assert result.dtype == torch.complex64
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)


def test_public_pytorch_logm_accepts_array_like_integer_inputs_when_active():
    if getattr(backend, "__backend_name__", None) != "pytorch":
        pytest.skip("public PyTorch backend is not active")

    result = backend.linalg.logm([[1, 0], [0, 1]])

    assert result.dtype.is_floating_point
    assert torch.allclose(result, torch.zeros_like(result))
