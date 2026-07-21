"""Regression tests for uint8 PyTorch assignment indices."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_raw_pytorch_assignment_treats_uint8_tensor_indices_as_integer_indices(
    backend_name,
):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        backend_name,
        """
import torch
import pyrecest  # noqa: F401 - triggers runtime backend compatibility patches
import pyrecest._backend.pytorch as raw_pytorch

indices = torch.tensor([1], dtype=torch.uint8)
assigned = raw_pytorch.assignment(raw_pytorch.array([0.0, 0.0, 0.0]), 7.0, indices)
accumulated = raw_pytorch.assignment_by_sum(raw_pytorch.array([0.0, 0.0, 0.0]), 2.5, indices)

assert raw_pytorch.to_numpy(assigned).tolist() == [0.0, 7.0, 0.0]
assert raw_pytorch.to_numpy(accumulated).tolist() == [0.0, 2.5, 0.0]
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_public_pytorch_assignment_treats_uint8_tensor_indices_as_integer_indices():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import torch
import pyrecest.backend as backend

indices = torch.tensor([1], dtype=torch.uint8)
assigned = backend.assignment(backend.array([0.0, 0.0, 0.0]), 7.0, indices)
accumulated = backend.assignment_by_sum(backend.array([0.0, 0.0, 0.0]), 2.5, indices)

assert backend.to_numpy(assigned).tolist() == [0.0, 7.0, 0.0]
assert backend.to_numpy(accumulated).tolist() == [0.0, 2.5, 0.0]
""",
    )

    assert result.returncode == 0, result.stderr
