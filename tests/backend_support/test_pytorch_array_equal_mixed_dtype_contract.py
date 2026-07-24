import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _mixed_dtype_contract_code(target_module):
    return f"""
import torch
import pyrecest.stability  # noqa: F401  # triggers backend compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch


target = {target_module}

# PyTorch cannot promote uint64 and int64, but array_equal must return a bool.
assert not target.array_equal(
    torch.tensor([2**63 - 1], dtype=torch.uint64),
    torch.tensor([2**63 - 2], dtype=torch.int64),
)
assert target.array_equal(
    torch.tensor([1], dtype=torch.uint64),
    torch.tensor([1], dtype=torch.int64),
)

# Promoting int64 to float32 would round away this one-unit difference.
assert not target.array_equal(
    torch.tensor([2**24 + 1], dtype=torch.int64),
    torch.tensor([float(2**24)], dtype=torch.float32),
)
assert target.array_equal(
    torch.tensor([2**24], dtype=torch.int64),
    torch.tensor([float(2**24)], dtype=torch.float32),
)

print("ok")
"""


def test_raw_pytorch_array_equal_handles_mixed_integral_dtypes_exactly():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("numpy", _mixed_dtype_contract_code("raw_pytorch"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_array_equal_handles_mixed_integral_dtypes_exactly():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _mixed_dtype_contract_code("backend"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
