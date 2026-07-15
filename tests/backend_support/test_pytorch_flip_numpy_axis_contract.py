"""Regression tests for NumPy-style PyTorch flip axes."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_flip_accepts_numpy_integer_axis_values():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import torch

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

values = backend.asarray([[0, 1, 2], [3, 4, 5]])
expected_axis_0 = [[3, 4, 5], [0, 1, 2]]
expected_axis_1 = [[2, 1, 0], [5, 4, 3]]
expected_both = [[5, 4, 3], [2, 1, 0]]

for flip in (backend.flip, raw_backend.flip):
    assert backend.to_numpy(flip(values, np.int64(0))).tolist() == expected_axis_0
    assert backend.to_numpy(flip(values, np.asarray(1))).tolist() == expected_axis_1
    assert backend.to_numpy(flip(values, np.asarray([0, 1]))).tolist() == expected_both
    assert backend.to_numpy(flip(values, torch.tensor(0))).tolist() == expected_axis_0
    assert backend.to_numpy(flip(values, torch.tensor([0, 1]))).tolist() == expected_both
""",
    )

    assert result.returncode == 0, result.stderr
