"""Regression tests for PyTorch log1p backend contract."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_log1p_accepts_numpy_style_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import math
import pyrecest.backend as backend

scalar = backend.log1p(1.0)
vector = backend.log1p([0.0, 1.0, 3.0])
out = backend.empty_like(vector)
returned = backend.log1p([0.0, 1.0, 3.0], out=out)

assert returned is out
assert abs(backend.to_numpy(scalar).item() - math.log1p(1.0)) < 1e-6
values = backend.to_numpy(vector).tolist()
out_values = backend.to_numpy(out).tolist()
for actual, expected in zip(values, [math.log1p(0.0), math.log1p(1.0), math.log1p(3.0)]):
    assert abs(actual - expected) < 1e-6
for actual, expected in zip(out_values, [math.log1p(0.0), math.log1p(1.0), math.log1p(3.0)]):
    assert abs(actual - expected) < 1e-6
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_raw_pytorch_log1p_accepts_array_like_inputs_under_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import math
import pyrecest._backend.pytorch as raw_pytorch

values = raw_pytorch.log1p([0.0, 1.0, 3.0])
as_numpy = raw_pytorch.to_numpy(values).tolist()

for actual, expected in zip(as_numpy, [math.log1p(0.0), math.log1p(1.0), math.log1p(3.0)]):
    assert abs(actual - expected) < 1e-6
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
