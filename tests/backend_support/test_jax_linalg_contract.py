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


@pytest.mark.backend_portable
def test_jax_linalg_accepts_array_like_inputs_directly():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    import jax.numpy as jnp
    from pyrecest._backend.jax import linalg

    assert abs(float(linalg.det([[1.0, 2.0], [3.0, 4.0]])) + 2.0) < 1e-5

    solution = linalg.solve([[2.0, 0.0], [0.0, 4.0]], [2.0, 8.0])
    assert bool(jnp.allclose(solution, jnp.array([1.0, 2.0])))

    inverse = linalg.inv([[1.0, 0.0], [0.0, 2.0]])
    assert bool(jnp.allclose(inverse, jnp.array([[1.0, 0.0], [0.0, 0.5]])))

    block = linalg.block_diag([[1.0]], [[2.0]])
    assert bool(jnp.allclose(block, jnp.array([[1.0, 0.0], [0.0, 2.0]])))

    exponential = linalg.expm([[0.0, 0.0], [0.0, 0.0]])
    assert bool(jnp.allclose(exponential, jnp.eye(2)))

    root = linalg.sqrtm([[4.0, 0.0], [0.0, 9.0]])
    assert bool(jnp.allclose(root, jnp.array([[2.0, 0.0], [0.0, 3.0]]), atol=1e-5))


@pytest.mark.backend_portable
def test_jax_matrix_power_accepts_zero_dimensional_integer_scalars():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "jax",
        """
import jax.numpy as jnp
import numpy as np
import pyrecest.backend as backend

matrix = backend.array([[1.0, 1.0], [0.0, 1.0]])
expected = backend.array([[1.0, 2.0], [0.0, 1.0]])

np_scalar_result = backend.linalg.matrix_power(matrix, np.array(2))
jax_scalar_result = backend.linalg.matrix_power(matrix, jnp.array(2))

assert bool(jnp.allclose(np_scalar_result, expected))
assert bool(jnp.allclose(jax_scalar_result, expected))
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_jax_qr_accepts_numpy_economic_mode():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "jax",
        """
import numpy as np
import pyrecest.backend as backend

values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
economic = backend.linalg.qr(backend.array(values), mode="economic")
expected = np.linalg.qr(values, mode="raw")[0].swapaxes(-1, -2)
np.testing.assert_allclose(np.asarray(economic), expected, rtol=1e-5, atol=1e-5)

batched_values = np.stack([values, values + 1.0])
batched_economic = backend.linalg.qr(backend.array(batched_values), mode="economic")
batched_expected = np.linalg.qr(batched_values, mode="raw")[0].swapaxes(-1, -2)
np.testing.assert_allclose(
    np.asarray(batched_economic),
    batched_expected,
    rtol=1e-5,
    atol=1e-5,
)
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
