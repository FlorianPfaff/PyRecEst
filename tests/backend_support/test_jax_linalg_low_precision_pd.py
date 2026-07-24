"""Regression coverage for low-precision JAX positive-definite checks."""

from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.backend_portable
def test_jax_pd_predicate_supports_low_precision_real_matrices():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    import jax.numpy as jnp
    from pyrecest._backend.jax import linalg

    positive_definite = [[2.0, 0.0], [0.0, 1.0]]
    indefinite = [[1.0, 2.0], [2.0, 1.0]]

    for dtype in (jnp.float16, jnp.bfloat16):
        assert bool(
            linalg.is_single_matrix_pd(jnp.asarray(positive_definite, dtype=dtype))
        )
        assert not bool(
            linalg.is_single_matrix_pd(jnp.asarray(indefinite, dtype=dtype))
        )
