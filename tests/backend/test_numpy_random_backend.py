import numpy as np
import pytest
from pyrecest._backend.numpy import random


def test_rand_accepts_backend_size_keyword():
    random.seed(0)

    samples = random.rand(size=(2, 3))

    assert samples.shape == (2, 3)
    assert samples.dtype == np.float64


def test_rand_keeps_numpy_positional_dimensions():
    random.seed(0)

    assert random.rand(2, 3).shape == (2, 3)
    assert random.rand(4).shape == (4,)


def test_rand_rejects_ambiguous_positional_and_size_arguments():
    with pytest.raises(TypeError, match="positional dimensions or size"):
        random.rand(2, size=(3,))
