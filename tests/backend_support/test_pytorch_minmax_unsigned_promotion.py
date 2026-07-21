"""Regression tests for mixed unsigned PyTorch extrema promotion."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _unsigned_promotion_code(target_module: str) -> str:
    return f"""
import numpy as np
import torch
import pyrecest  # noqa: F401  # triggers backend-support compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch


target = {target_module}
cases = (
    (
        np.array([0, 65535], dtype=np.uint16),
        np.array([-1, 32767], dtype=np.int16),
    ),
    (
        np.array([1, 2**32 - 1], dtype=np.uint32),
        np.array([-1, 2**31 - 1], dtype=np.int32),
    ),
    (
        np.array([0, 2**63], dtype=np.uint64),
        np.array([-1, 2**63 - 1], dtype=np.int64),
    ),
)

for left_numpy, right_numpy in cases:
    left = torch.from_numpy(left_numpy)
    right = torch.from_numpy(right_numpy)

    maximum_result = target.maximum(left, right)
    maximum_expected = torch.from_numpy(np.maximum(left_numpy, right_numpy))
    assert maximum_result.dtype == maximum_expected.dtype
    assert torch.equal(maximum_result, maximum_expected)

    minimum_result = target.minimum(left, right)
    minimum_expected = torch.from_numpy(np.minimum(left_numpy, right_numpy))
    assert minimum_result.dtype == minimum_expected.dtype
    assert torch.equal(minimum_result, minimum_expected)

print("ok")
"""


def test_raw_pytorch_minmax_matches_numpy_unsigned_promotion():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("numpy", _unsigned_promotion_code("raw_pytorch"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_minmax_matches_numpy_unsigned_promotion():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _unsigned_promotion_code("backend"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
