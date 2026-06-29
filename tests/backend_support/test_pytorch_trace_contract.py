import importlib.util

import pytest


def test_raw_pytorch_trace_accepts_numpy_signature():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
    import torch
    import pyrecest._backend.pytorch as raw_pytorch_backend

    values = torch.arange(12.0, dtype=torch.float64).reshape(2, 2, 3)
    expected = torch.tensor([6.0, 18.0], dtype=torch.float64)

    result = raw_pytorch_backend.trace(values, offset=1, axis1=-2, axis2=-1)
    assert torch.allclose(result, expected)

    out = torch.empty(2, dtype=torch.float64)
    returned = raw_pytorch_backend.trace(
        values,
        offset=1,
        axis1=-2,
        axis2=-1,
        dtype=torch.float64,
        out=out,
    )
    assert returned is out
    assert torch.allclose(out, expected)

    scalar_result = raw_pytorch_backend.trace([[1.0, 2.0], [3.0, 4.0]])
    assert tuple(scalar_result.shape) == ()
    assert float(scalar_result) == pytest.approx(5.0)
