"""Regression tests for PyTorch maximum/minimum device preservation."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _device_contract_code(target_module: str) -> str:
    return f"""
import torch
import pyrecest  # noqa: F401  # triggers backend-support compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch


def _non_cpu_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("meta")


target = {target_module}
device = _non_cpu_device()
right = torch.tensor([1.0, 4.0], device=device)

maximum_result = target.maximum([2.0, 3.0], right)
assert maximum_result.device.type == device.type
assert tuple(maximum_result.shape) == (2,)
if device.type != "meta":
    assert torch.allclose(maximum_result.cpu(), torch.tensor([2.0, 4.0]))

minimum_result = target.minimum(torch.tensor([2.0, 3.0]), right)
assert minimum_result.device.type == device.type
assert tuple(minimum_result.shape) == (2,)
if device.type != "meta":
    assert torch.allclose(minimum_result.cpu(), torch.tensor([1.0, 3.0]))

left = torch.tensor([2.0, 3.0], device=device)
minimum_arraylike_result = target.minimum(left, [1.0, 4.0])
assert minimum_arraylike_result.device.type == device.type
assert tuple(minimum_arraylike_result.shape) == (2,)
if device.type != "meta":
    assert torch.allclose(minimum_arraylike_result.cpu(), torch.tensor([1.0, 3.0]))

print("ok")
"""


def test_raw_pytorch_maximum_minimum_prefer_existing_non_cpu_device_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("numpy", _device_contract_code("raw_pytorch"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_maximum_minimum_prefer_existing_non_cpu_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _device_contract_code("backend"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
