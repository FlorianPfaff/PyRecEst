"""Regression tests for PyTorch backend arg reduction keyword compatibility."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_arg_reductions_accept_numpy_axis_and_keepdims_keywords():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

values = backend.asarray([[1.0, 5.0, 0.0], [4.0, 2.0, 3.0], [-1.0, 3.0, 6.0]])
assert backend.to_numpy(backend.argmax(values, axis=1)).tolist() == [1, 0, 2]
assert backend.to_numpy(backend.argmin(values, axis=0, keepdims=True)).tolist() == [
    [2, 1, 0]
]
assert backend.to_numpy(backend.argmax(values, dim=0, keepdim=True)).tolist() == [
    [1, 0, 2]
]

flags = backend.asarray([[False, True, True], [True, True, False]])
assert backend.to_numpy(backend.argmax(flags, axis=1)).tolist() == [1, 0]
assert backend.to_numpy(backend.argmin(flags, axis=1)).tolist() == [0, 2]
assert backend.to_numpy(backend.argmax(flags, keepdims=True)).tolist() == [[1]]
""",
    )

    assert result.returncode == 0, result.stderr
