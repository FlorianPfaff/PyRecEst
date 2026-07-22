"""Regression coverage for PyTorch sparse reconstruction semantics."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable

EXPECTED_VECTOR = [5, 0, 3, 0]
EXPECTED_MATRIX = [[0.0, 7.0], [4.0, 0.0]]


def test_raw_pytorch_array_from_sparse_duplicate_indices_match_numpy_put():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = f"""
import pyrecest  # noqa: F401  # triggers raw-backend runtime patches
import torch
import pyrecest._backend.pytorch as raw_pytorch_backend

vector = raw_pytorch_backend.array_from_sparse(
    [(0,), (0,), (2,)],
    torch.tensor([1, 5, 3], dtype=torch.int64),
    (4,),
)
assert vector.tolist() == {EXPECTED_VECTOR!r}

matrix = raw_pytorch_backend.array_from_sparse(
    [(0, 1), (0, 1), (1, 0)],
    [2.0, 7.0, 4.0],
    (2, 2),
)
assert matrix.tolist() == {EXPECTED_MATRIX!r}
print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_array_from_sparse_duplicate_indices_match_numpy_put():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = f"""
import pyrecest.backend as backend

vector = backend.array_from_sparse(
    [(0,), (0,), (2,)],
    backend.array([1, 5, 3], dtype=backend.int64),
    (4,),
)
assert backend.to_numpy(vector).tolist() == {EXPECTED_VECTOR!r}

matrix = backend.array_from_sparse(
    [(0, 1), (0, 1), (1, 0)],
    [2.0, 7.0, 4.0],
    (2, 2),
)
assert backend.to_numpy(matrix).tolist() == {EXPECTED_MATRIX!r}
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
