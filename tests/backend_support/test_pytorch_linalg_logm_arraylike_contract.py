"""Regression tests for PyTorch linalg matrix logarithm input coercion."""

import pytest
from tests.support.backend_runner import run_backend_code


def test_pytorch_linalg_logm_accepts_array_like_integer_matrix():
    pytest.importorskip("torch")

    code = """
import torch

from pyrecest.backend import linalg

result = linalg.logm([[1, 0], [0, 1]])
expected = torch.zeros((2, 2), dtype=result.dtype)

assert torch.is_tensor(result)
assert result.dtype.is_floating_point
assert torch.allclose(result, expected)
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
