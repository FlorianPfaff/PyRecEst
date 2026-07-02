"""Regression tests for raw PyTorch backend comparison helpers."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_raw_pytorch_comparisons_accept_numpy_style_array_like_inputs(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        backend_name,
        """
import importlib

raw_pytorch = importlib.import_module("pyrecest._backend.pytorch")

right = raw_pytorch.asarray([0, 2, 4])
greater_result = raw_pytorch.greater([1, 2, 3], right)
less_result = raw_pytorch.less(right, [1, 2, 3])
logical_result = raw_pytorch.logical_or([True, False], raw_pytorch.asarray([False, True]))

assert raw_pytorch.to_numpy(greater_result).tolist() == [True, False, False]
assert raw_pytorch.to_numpy(less_result).tolist() == [True, False, False]
assert raw_pytorch.to_numpy(logical_result).tolist() == [True, True]
""",
    )

    assert result.returncode == 0, result.stderr
