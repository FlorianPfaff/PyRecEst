"""Regression tests for backend random.uniform bounds handling."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

_UNIFORM_ARRAY_BOUNDS_CHECK = """
import pyrecest.backend as backend
from pyrecest.backend import random

low = backend.array([0.0, 10.0])
high = backend.array([1.0, 20.0])

sample = random.uniform(low=low, high=high, size=(2,))

assert sample.shape == (2,)
assert bool(backend.all(sample >= low))
assert bool(backend.all(sample <= high))
print("ok")
"""


_UNIFORM_ARRAY_BOUNDS_REJECTION_CHECK = """
import pyrecest.backend as backend
from pyrecest.backend import random

low = backend.array([0.0, 2.0])
high = backend.array([1.0, 1.0])

try:
    random.uniform(low=low, high=high, size=(2,))
except ValueError as exc:
    assert "Upper bound" in str(exc)
else:
    raise AssertionError("array-valued invalid uniform bounds were accepted")

print("ok")
"""


@pytest.mark.backend_portable
@pytest.mark.parametrize(
    "backend,required_module",
    [
        ("jax", "jax"),
        ("pytorch", "torch"),
    ],
)
def test_uniform_accepts_array_valued_bounds(backend, required_module):
    if importlib.util.find_spec(required_module) is None:
        pytest.skip(f"{backend} backend dependency is not installed")

    result = run_backend_code(backend, _UNIFORM_ARRAY_BOUNDS_CHECK)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
@pytest.mark.parametrize(
    "backend,required_module",
    [
        ("jax", "jax"),
        ("pytorch", "torch"),
    ],
)
def test_uniform_rejects_array_valued_invalid_bounds(backend, required_module):
    if importlib.util.find_spec(required_module) is None:
        pytest.skip(f"{backend} backend dependency is not installed")

    result = run_backend_code(backend, _UNIFORM_ARRAY_BOUNDS_REJECTION_CHECK)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
