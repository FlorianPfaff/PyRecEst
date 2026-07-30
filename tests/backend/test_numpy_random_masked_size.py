import numpy as np
import pytest
from pyrecest._backend.numpy import random

_MASKED_SIZE_ARGUMENTS = (
    np.ma.masked,
    np.ma.array(3, mask=True),
    np.ma.array([2, 3], mask=[False, True]),
    (2, np.ma.masked),
)


@pytest.mark.parametrize("bad_size", _MASKED_SIZE_ARGUMENTS)
@pytest.mark.parametrize(
    "sampler",
    (
        lambda size: random.rand(size=size),
        lambda size: random.uniform(size=size),
        lambda size: random.normal(size=size),
        lambda size: random.multivariate_normal([0.0], [[1.0]], size=size),
        lambda size: random.choice(np.arange(5), size=size),
        lambda size: random.randint(0, 5, size=size),
        lambda size: random.multinomial(2, [0.25, 0.75], size=size),
    ),
)
def test_numpy_random_rejects_masked_size_arguments(sampler, bad_size):
    with pytest.raises(TypeError, match="size must be None"):
        sampler(bad_size)


def test_numpy_random_accepts_fully_unmasked_masked_array_size():
    samples = random.rand(size=np.ma.array([2, 3], mask=False))

    assert samples.shape == (2, 3)
