"""Regression tests for PyTorch where device preservation."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _where_device_contract_code(target_module: str) -> str:
    return f"""
import torch
import pyrecest  # noqa: F401  # triggers backend-support compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch



def _non_cpu_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("meta")



def _assert_one_sided_where_raises(*args, **kwargs):
    try:
        target.where(*args, **kwargs)
    except ValueError as exc:
        assert "either both or neither" in str(exc)
    else:
        raise AssertionError("one-sided where did not raise ValueError")


target = {target_module}
device = _non_cpu_device()

right = torch.ones(2, device=device)
result = target.where([True, False], torch.tensor([1.0, 2.0]), right)
assert result.device.type == device.type
assert tuple(result.shape) == (2,)
if device.type != "meta":
    assert torch.allclose(result.cpu(), torch.tensor([1.0, 1.0]))

condition = torch.tensor([True, False], device=device)
result = target.where(condition, [1.0, 2.0], torch.tensor([3.0, 4.0]))
assert result.device.type == device.type
assert tuple(result.shape) == (2,)
if device.type != "meta":
    assert torch.allclose(result.cpu(), torch.tensor([1.0, 4.0]))

_assert_one_sided_where_raises([True, False], [1.0, 2.0])
_assert_one_sided_where_raises([True, False], y=[1.0, 2.0])

print("ok")
"""


def test_raw_pytorch_where_prefers_existing_non_cpu_device_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("numpy", _where_device_contract_code("raw_pytorch"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_where_prefers_existing_non_cpu_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _where_device_contract_code("backend"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
