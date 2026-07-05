"""Regression tests for PyTorch unique return-value flags."""

from __future__ import annotations

import pytest

from tests.support.backend_runner import run_backend_code


def test_raw_pytorch_unique_accepts_return_counts_under_numpy_backend():
    pytest.importorskip("torch")

    code = """
import pyrecest
import pyrecest._backend.pytorch as pytorch_backend

values, counts = pytorch_backend.unique([2, 1, 2, 3, 1], return_counts=True)

assert values.tolist() == [1, 2, 3]
assert counts.tolist() == [2, 2, 1]
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr


def test_public_pytorch_unique_accepts_return_inverse_and_counts():
    pytest.importorskip("torch")

    code = """
import pyrecest.backend as backend

values, inverse, counts = backend.unique(
    [2, 1, 2, 3, 1],
    return_inverse=True,
    return_counts=True,
)

assert values.tolist() == [1, 2, 3]
assert inverse.tolist() == [1, 0, 1, 2, 0]
assert counts.tolist() == [2, 2, 1]
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
