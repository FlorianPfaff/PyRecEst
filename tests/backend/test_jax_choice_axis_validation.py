import pytest

pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import random  # noqa: E402


def test_choice_rejects_bool_axis_for_arrays():
    values = jnp.array([[0, 1, 2], [3, 4, 5]])

    with pytest.raises(TypeError):
        random.choice(values, size=1, axis=True)
