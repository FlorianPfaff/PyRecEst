"""Regression tests for PyTorch apply_along_axis callback arguments."""

import pytest
from tests.support.backend_runner import run_backend_code


def test_public_pytorch_apply_along_axis_forwards_callback_arguments():
    pytest.importorskip("torch")
    code = r"""
import torch
import pyrecest.backend as backend


def affine(row, scale, *, offset):
    return row * scale + offset


values = torch.arange(6, dtype=torch.float64).reshape(2, 3)
result = backend.apply_along_axis(affine, 1, values, 2.0, offset=1.0)
expected = values * 2.0 + 1.0
assert torch.equal(result, expected)
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr


def test_raw_pytorch_apply_along_axis_forwards_callback_arguments_after_import():
    pytest.importorskip("torch")
    code = r"""
import torch
import pyrecest  # noqa: F401 - activates backend-support compatibility patches
import pyrecest._backend.pytorch as raw_pytorch


def affine(row, scale, *, offset):
    return row * scale + offset


values = torch.arange(6, dtype=torch.float64).reshape(2, 3)
result = raw_pytorch.apply_along_axis(affine, 1, values, 2.0, offset=1.0)
expected = values * 2.0 + 1.0
assert torch.equal(result, expected)
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
