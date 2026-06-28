import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import random  # noqa: E402


@pytest.mark.parametrize(
    "pvals",
    [
        [True, False],
        np.array([True, False]),
        jnp.array([True, False]),
    ],
)
def test_multinomial_rejects_boolean_pvals(pvals):
    with pytest.raises(TypeError, match="real numeric"):
        random.multinomial(1, pvals)


@pytest.mark.parametrize(
    "pvals",
    [
        [1.0 + 0.0j],
        np.array([1.0 + 0.0j]),
        jnp.array([1.0 + 0.0j]),
    ],
)
def test_multinomial_rejects_complex_pvals(pvals):
    with pytest.raises(TypeError, match="real numeric"):
        random.multinomial(1, pvals)


def test_multinomial_accepts_real_pvals():
    random.seed(0)

    result = random.multinomial(2, [1.0, 0.0])

    assert result.shape == (2,)
    assert int(result.sum()) == 2
