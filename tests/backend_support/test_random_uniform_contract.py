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


_UNIFORM_NONFINITE_BOUNDS_REJECTION_CHECK = """
from pyrecest.backend import random

calls = [
    ("nan low", lambda: random.uniform(float("nan"), 1.0, size=(2,))),
    ("nan high", lambda: random.uniform(0.0, float("nan"), size=(2,))),
    ("-inf low", lambda: random.uniform(float("-inf"), 1.0, size=(2,))),
    ("inf high", lambda: random.uniform(0.0, float("inf"), size=(2,))),
]

for label, call in calls:
    try:
        call()
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError(f"uniform accepted {label}")

print("ok")
"""


_UNIFORM_OVERFLOWING_RANGE_REJECTION_CHECK = """
import numpy as np
from pyrecest.backend import random

maximum = np.finfo(np.float64).max
try:
    random.uniform(low=-maximum, high=maximum, size=(2,))
except OverflowError as exc:
    assert "high - low range exceeds valid bounds" in str(exc)
else:
    raise AssertionError("uniform accepted a non-representable finite range")

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


@pytest.mark.backend_portable
@pytest.mark.parametrize(
    "backend,required_module",
    [
        ("numpy", "numpy"),
        ("jax", "jax"),
        ("pytorch", "torch"),
    ],
)
def test_uniform_rejects_nonfinite_bounds(backend, required_module):
    if importlib.util.find_spec(required_module) is None:
        pytest.skip(f"{backend} backend dependency is not installed")

    result = run_backend_code(backend, _UNIFORM_NONFINITE_BOUNDS_REJECTION_CHECK)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
@pytest.mark.parametrize(
    "backend,required_module",
    [
        ("numpy", "numpy"),
        ("pytorch", "torch"),
    ],
)
def test_uniform_rejects_overflowing_finite_range(backend, required_module):
    if importlib.util.find_spec(required_module) is None:
        pytest.skip(f"{backend} backend dependency is not installed")

    result = run_backend_code(backend, _UNIFORM_OVERFLOWING_RANGE_REJECTION_CHECK)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
