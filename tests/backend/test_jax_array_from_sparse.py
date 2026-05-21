import pytest

pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import array_from_sparse  # noqa: E402


def test_array_from_sparse_uses_flat_indices_for_multidimensional_target():
    indices = jnp.array([[0, 0], [1, 2]])
    data = jnp.array([3.0, 5.0])

    dense = array_from_sparse(indices, data, (2, 3))

    expected = jnp.array([[3.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    assert jnp.array_equal(dense, expected)


def test_array_from_sparse_handles_non_first_axis_flat_positions():
    indices = jnp.array([[0, 1, 2], [1, 0, 3]])
    data = jnp.array([7, 11])

    dense = array_from_sparse(indices, data, (2, 2, 4))

    expected = jnp.zeros((2, 2, 4), dtype=data.dtype)
    expected = expected.at[0, 1, 2].set(7)
    expected = expected.at[1, 0, 3].set(11)
    assert jnp.array_equal(dense, expected)
