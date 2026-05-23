"""Regression tests for JAX backend linear-algebra helpers."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_jax_is_single_matrix_pd_is_supported():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "jax",
        """
import pyrecest.backend as backend

real_pd = backend.array([[2.0, 0.0], [0.0, 1.0]])
real_indefinite = backend.array([[1.0, 2.0], [2.0, 1.0]])
non_square = backend.ones((2, 3))
complex_hermitian_pd = backend.array([[2.0 + 0.0j, 0.5j], [-0.5j, 2.0 + 0.0j]])
complex_non_hermitian = backend.array([[1.0 + 0.0j, 1.0j], [1.0j, 1.0 + 0.0j]])

assert bool(backend.linalg.is_single_matrix_pd(real_pd)) is True
assert bool(backend.linalg.is_single_matrix_pd(real_indefinite)) is False
assert backend.linalg.is_single_matrix_pd(non_square) is False
assert bool(backend.linalg.is_single_matrix_pd(complex_hermitian_pd)) is True
assert bool(backend.linalg.is_single_matrix_pd(complex_non_hermitian)) is False
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
