"""Regression coverage for JAX sparse reconstruction semantics."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable

EXPECTED_VECTOR = [5, 0, 7, 0]


def test_public_jax_array_from_sparse_accepts_flat_1d_indices():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    code = f"""
import pyrecest.backend as backend

vector = backend.array_from_sparse(
    [0, 2],
    backend.array([5, 7], dtype=backend.int32),
    (4,),
)
assert backend.to_numpy(vector).tolist() == {EXPECTED_VECTOR!r}
print("ok")
"""
    result = run_backend_code("jax", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_raw_jax_array_from_sparse_accepts_flat_1d_indices():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    code = f"""
import importlib

from pyrecest.backend import to_numpy

raw_jax = importlib.import_module("pyrecest._backend.jax")
vector = raw_jax.array_from_sparse(
    [0, 2],
    raw_jax.array([5, 7], dtype=raw_jax.int32),
    (4,),
)
assert to_numpy(vector).tolist() == {EXPECTED_VECTOR!r}
print("ok")
"""
    result = run_backend_code("jax", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
