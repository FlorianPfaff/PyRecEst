import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import random  # noqa: E402


def test_choice_without_replacement_shuffle_false_preserves_order():
    values = jnp.array([10, 20, 30, 40, 50])
    matrix = jnp.array([[10, 20, 30], [40, 50, 60]])

    random.seed(0)
    samples = random.choice(values, size=values.shape[0], replace=False, shuffle=False)
    column_samples = random.choice(
        matrix,
        size=matrix.shape[1],
        replace=False,
        axis=1,
        shuffle=False,
    )

    assert jnp.array_equal(samples, values)
    assert jnp.array_equal(column_samples, matrix)


@pytest.mark.parametrize("shuffle", [np.array(False), jnp.array(False)])
def test_choice_accepts_scalar_boolean_shuffle_flag(shuffle):
    random.seed(0)

    samples = random.choice(
        jnp.array([0, 1, 2]), size=3, replace=False, shuffle=shuffle
    )

    assert samples.shape == (3,)


@pytest.mark.parametrize("bad_shuffle", ["False", "True", 1, 0, None, np.array([True])])
def test_choice_rejects_non_boolean_shuffle_flag(bad_shuffle):
    with pytest.raises(TypeError, match="shuffle must be a boolean"):
        random.choice(jnp.array([0, 1, 2]), size=2, shuffle=bad_shuffle)
