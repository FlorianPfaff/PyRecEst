"""Regression tests for PyTorch arctan backend contract."""

from __future__ import annotations

import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_arctan_accepts_numpy_style_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import math
import pyrecest.backend as backend

scalar = backend.arctan(1.0)
vector = backend.arctan([0.0, 1.0])

assert abs(backend.to_numpy(scalar).item() - math.pi / 4.0) < 1e-6
values = backend.to_numpy(vector).tolist()
assert abs(values[0] - 0.0) < 1e-12
assert abs(values[1] - math.pi / 4.0) < 1e-6
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_raw_pytorch_arctan_accepts_array_like_inputs_under_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import math
import pyrecest._backend.pytorch as raw_pytorch

values = raw_pytorch.arctan([0.0, 1.0])
as_numpy = raw_pytorch.to_numpy(values).tolist()

assert abs(as_numpy[0] - 0.0) < 1e-12
assert abs(as_numpy[1] - math.pi / 4.0) < 1e-6
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
