"""Regression tests for empty random.uniform bounds across backends."""

from __future__ import annotations

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


_CHECK = """
import pyrecest.backend as backend
from pyrecest.backend import random

samples = random.uniform(2.0, 1.0, size=(0, 3))
assert tuple(backend.shape(samples)) == (0, 3)

try:
    random.uniform(2.0, 1.0, size=(3,))
except ValueError as exc:
    assert "Upper bound" in str(exc)
else:
    raise AssertionError("non-empty descending interval was accepted")

print("ok")
"""


@pytest.mark.parametrize("backend_name", ["numpy", "pytorch", "jax"])
def test_uniform_accepts_empty_descending_interval(backend_name):
    if backend_name == "pytorch":
        pytest.importorskip("torch")
    elif backend_name == "jax":
        pytest.importorskip("jax")

    result = run_backend_code(backend_name, _CHECK)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
