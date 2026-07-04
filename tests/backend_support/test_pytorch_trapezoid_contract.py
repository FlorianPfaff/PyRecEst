"""Regression tests for PyTorch trapezoid backend compatibility."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_trapezoid_accepts_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

scalar = backend.trapezoid([1.0, 2.0, 4.0])
assert float(backend.to_numpy(scalar)) == 4.5

weighted = backend.trapezoid(
    [[1.0, 2.0, 4.0], [2.0, 3.0, 5.0]],
    x=[0.0, 0.5, 2.0],
    axis=1,
)
assert backend.to_numpy(weighted).tolist() == [5.25, 7.25]
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_raw_pytorch_trapezoid_accepts_array_like_inputs_after_pyrecest_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import pyrecest  # noqa: F401
import pyrecest._backend.pytorch as raw_pytorch

scalar = raw_pytorch.trapezoid([1.0, 2.0, 4.0])
assert float(raw_pytorch.to_numpy(scalar)) == 4.5
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
