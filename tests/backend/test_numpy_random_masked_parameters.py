import numpy as np
import pytest
from pyrecest._backend.numpy import random


@pytest.mark.parametrize(
    ("sampler", "message"),
    (
        (
            lambda: random.randint(np.ma.array(5, mask=True)),
            "high must contain integer values",
        ),
        (
            lambda: random.randint(np.ma.array(1, mask=True), 5),
            "low must contain integer values",
        ),
        (
            lambda: random.randint(0, np.ma.array(5, mask=True)),
            "high must contain integer values",
        ),
        (
            lambda: random.multinomial(
                np.ma.array(2, mask=True),
                [0.25, 0.75],
            ),
            "n must be a non-negative integer",
        ),
        (
            lambda: random.multinomial(
                2,
                np.ma.array([0.25, 0.75], mask=[False, True]),
            ),
            "pvals must be real numeric",
        ),
        (
            lambda: random.multinomial(2, [0.25, np.ma.masked]),
            "pvals must be real numeric",
        ),
    ),
)
def test_numpy_random_rejects_masked_sampling_parameters(sampler, message):
    with pytest.raises(TypeError, match=message):
        sampler()


def test_numpy_random_accepts_fully_unmasked_masked_parameters():
    samples = random.randint(
        np.ma.array(1, mask=False),
        np.ma.array(4, mask=False),
        size=8,
    )
    counts = random.multinomial(
        np.ma.array(3, mask=False),
        np.ma.array([0.25, 0.75], mask=False),
    )

    assert samples.shape == (8,)
    assert np.all((samples >= 1) & (samples < 4))
    assert counts.shape == (2,)
    assert counts.sum() == 3
