import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import random  # noqa: E402


def test_multivariate_normal_accepts_numpy_argument_order():
    random.seed(0)
    mean = jnp.array([1.0, -1.0])
    cov = jnp.eye(2)

    assert random.multivariate_normal(mean, cov).shape == (2,)
    assert random.multivariate_normal(mean, cov, 3).shape == (3, 2)
    assert random.multivariate_normal(mean, cov, (4,)).shape == (4, 2)


def test_multivariate_normal_accepts_shape_keyword():
    random.seed(0)
    mean = jnp.array([1.0, -1.0])
    cov = jnp.eye(2)

    assert random.multivariate_normal(mean, cov, shape=(5,)).shape == (5, 2)

    with pytest.raises(TypeError):
        random.multivariate_normal(mean, cov, size=(1,), shape=(2,))
