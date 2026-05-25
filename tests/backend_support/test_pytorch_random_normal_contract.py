"""Regression tests for PyTorch backend random.normal keyword compatibility."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

_ARRAY_PARAMETER_CHECK = """
import pyrecest.backend as backend
from pyrecest.backend import random

backend.random.seed(7)
loc = backend.asarray([1.0, 2.0])
scale = backend.asarray([0.1, 0.2])

sample = random.normal(loc=loc, scale=scale)
list_sample = random.normal(loc=[1.0, 2.0], scale=0.1, size=(3, 2))

assert sample.shape == (2,)
assert list_sample.shape == (3, 2)
assert backend.to_numpy(sample).dtype.kind == "f"
assert backend.to_numpy(list_sample).dtype.kind == "f"
print("ok")
"""


_ARRAY_PARAMETER_NEGATIVE_SCALE_CHECK = """
from pyrecest.backend import random

try:
    random.normal(loc=[1.0, 2.0], scale=-1.0, size=(2,))
except ValueError as exc:
    assert "scale" in str(exc)
else:
    raise AssertionError("array-valued normal accepted a negative scale")

print("ok")
"""


@pytest.mark.backend_portable
def test_pytorch_normal_accepts_array_valued_parameters():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _ARRAY_PARAMETER_CHECK)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_pytorch_normal_rejects_negative_array_scale_with_size():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _ARRAY_PARAMETER_NEGATIVE_SCALE_CHECK)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
