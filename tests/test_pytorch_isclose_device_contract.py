import pytest
from tests.support.backend_runner import run_backend_code


def test_pytorch_isclose_places_array_like_operands_on_existing_device():
    pytest.importorskip("torch")

    code = """
import torch
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

probe = torch.ones(2, device="meta")
for helper in (backend.isclose, raw_pytorch.isclose):
    result = helper(probe, [1.0, 1.0])
    assert result.device.type == "meta"
    assert tuple(result.shape) == (2,)
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
