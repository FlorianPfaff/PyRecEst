import numpy as np
import pytest
from pyrecest._backend.numpy import random

_INVALID_SIZE_ARGUMENTS = (
    True,
    np.bool_(True),
    np.array(True),
    (2, True),
    1.5,
    (2, 1.5),
    "3",
)


@pytest.mark.parametrize("bad_size", _INVALID_SIZE_ARGUMENTS)
@pytest.mark.parametrize(
    "sampler",
    (
        lambda size: random.randint(0, 5, size=size),
        lambda size: random.multinomial(2, [0.25, 0.75], size=size),
    ),
)
def test_numpy_random_rejects_invalid_size_arguments_with_backend_message(
    bad_size, sampler
):
    with pytest.raises(TypeError, match="size must be None"):
        sampler(bad_size)


@pytest.mark.parametrize(
    "sampler",
    (
        lambda size: random.randint(0, 5, size=size),
        lambda size: random.multinomial(2, [0.25, 0.75], size=size),
    ),
)
def test_numpy_random_rejects_negative_size_dimensions(sampler):
    with pytest.raises(ValueError, match="size dimensions must be non-negative"):
        sampler((2, -1))


def test_numpy_randint_normalizes_array_like_size_arguments():
    samples = random.randint(0, 5, size=np.array([2, 3]))

    assert samples.shape == (2, 3)


def test_numpy_multinomial_normalizes_array_like_size_arguments():
    samples = random.multinomial(2, [0.25, 0.75], size=np.array([2, 3]))

    assert samples.shape == (2, 3, 2)
