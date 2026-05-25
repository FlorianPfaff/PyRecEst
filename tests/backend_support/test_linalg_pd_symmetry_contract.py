"""Regression tests for positive-definite matrix predicates."""

from __future__ import annotations

import importlib.util

import pytest

import pyrecest.backend as backend
from tests.support.backend_runner import run_backend_code


_NONSYMMETRIC_CHOLESKY_ACCEPTED = [[1.0, 100.0], [0.0, 1.0]]


def _as_bool(value) -> bool:
    value = backend.to_numpy(value)
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


@pytest.mark.backend_portable
def test_is_single_matrix_pd_rejects_real_nonsymmetric_matrix():
    """PD predicates must reject real matrices outside the symmetric cone."""
    matrix = backend.array(_NONSYMMETRIC_CHOLESKY_ACCEPTED)

    assert _as_bool(backend.linalg.is_single_matrix_pd(matrix)) is False


@pytest.mark.backend_portable
def test_jax_is_single_matrix_pd_rejects_real_nonsymmetric_matrix():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "jax",
        """
import pyrecest.backend as backend

matrix = backend.array([[1.0, 100.0], [0.0, 1.0]])
assert bool(backend.linalg.is_single_matrix_pd(matrix)) is False
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_pytorch_is_single_matrix_pd_rejects_real_nonsymmetric_matrix():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

matrix = backend.array([[1.0, 100.0], [0.0, 1.0]])
assert bool(backend.linalg.is_single_matrix_pd(matrix)) is False
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
