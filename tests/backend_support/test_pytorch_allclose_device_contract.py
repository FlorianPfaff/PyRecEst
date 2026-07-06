"""Regression tests for the explicit PyTorch close-helper compatibility hook."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_close_hook_preserves_equal_nan_for_isclose():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import pyrecest._backend.pytorch as raw_pytorch
from pyrecest.backend_support._pytorch_allclose_device_contract import (
    patch_pytorch_allclose_device_contract,
)

patch_pytorch_allclose_device_contract()

left = [float("nan"), 1.0]
right = raw_pytorch.array([float("nan"), 1.0])

assert raw_pytorch.allclose(left, right, equal_nan=True)
isclose_result = raw_pytorch.isclose(left, right, equal_nan=True)

assert raw_pytorch.to_numpy(isclose_result).tolist() == [True, True]
assert getattr(raw_pytorch.isclose, "_pyrecest_device_contract", False)
assert getattr(raw_pytorch.isclose, "_pyrecest_missing_value_contract", False)
assert getattr(raw_pytorch.allclose, "_pyrecest_missing_value_contract", False)
""",
    )

    assert result.returncode == 0, result.stderr
