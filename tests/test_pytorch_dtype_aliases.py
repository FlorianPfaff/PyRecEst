import numpy as np
import pyrecest.backend as backend
import pytest


def test_pytorch_as_dtype_accepts_dtype_like_aliases():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific dtype alias regression test")

    torch = pytest.importorskip("torch")

    original_dtype = backend.get_default_dtype()
    try:
        assert backend.as_dtype(torch.float64) == torch.float64
        assert backend.as_dtype(np.float64) == torch.float64
        assert backend.as_dtype(np.dtype("complex128")) == torch.complex128
        assert backend.as_dtype("torch.float32") == torch.float32

        assert backend.set_default_dtype(np.dtype("float32")) == torch.float32
        assert backend.get_default_dtype() == torch.float32
        assert backend.get_default_cdtype() == torch.complex64

        assert backend.set_default_dtype("torch.float64") == torch.float64
        assert backend.get_default_dtype() == torch.float64
        assert backend.get_default_cdtype() == torch.complex128
    finally:
        backend.set_default_dtype(original_dtype)
