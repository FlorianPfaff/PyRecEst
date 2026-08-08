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


@pytest.mark.parametrize(
    "sampler",
    (
        lambda: random.uniform(np.ma.array(-5.0, mask=True), 1.0),
        lambda: random.uniform(0.0, np.ma.array(5.0, mask=True)),
        lambda: random.normal(loc=np.ma.array(7.0, mask=True)),
        lambda: random.normal(scale=np.ma.array(2.0, mask=True)),
        lambda: random.multivariate_normal(
            np.ma.array([0.0, 9.0], mask=[False, True]),
            np.eye(2),
        ),
        lambda: random.multivariate_normal(
            np.zeros(2),
            np.ma.array(
                np.eye(2),
                mask=[[False, False], [False, True]],
            ),
        ),
        lambda: random.choice(
            np.ma.array([10, 20], mask=[False, True]),
        ),
        lambda: random.choice(
            np.arange(2),
            p=np.ma.array([0.5, 0.5], mask=[False, True]),
        ),
        lambda: random.choice(
            np.arange(2),
            replace=np.ma.array(True, mask=True),
        ),
        lambda: random.choice(
            np.arange(2),
            shuffle=np.ma.array(True, mask=True),
        ),
        lambda: random.choice(
            np.arange(2),
            axis=np.ma.array(0, mask=True),
        ),
    ),
)
def test_numpy_random_rejects_masked_distribution_parameters(sampler):
    with pytest.raises(ValueError, match="masked"):
        sampler()


def test_numpy_random_accepts_clear_mask_distribution_parameters():
    random.seed(0)

    uniform_samples = random.uniform(
        np.ma.array(0.0, mask=False),
        np.ma.array(1.0, mask=False),
        size=2,
    )
    normal_samples = random.normal(
        loc=np.ma.array(0.0, mask=False),
        scale=np.ma.array(1.0, mask=False),
        size=2,
    )
    multivariate_samples = random.multivariate_normal(
        np.ma.array([0.0, 0.0], mask=False),
        np.ma.array(np.eye(2), mask=False),
        size=2,
    )
    choice_samples = random.choice(
        np.ma.array([10, 20], mask=False),
        size=2,
        replace=np.ma.array(True, mask=False),
        p=np.ma.array([0.25, 0.75], mask=False),
        axis=np.ma.array(0, mask=False),
        shuffle=np.ma.array(True, mask=False),
    )

    assert uniform_samples.shape == (2,)
    assert normal_samples.shape == (2,)
    assert multivariate_samples.shape == (2, 2)
    assert choice_samples.shape == (2,)
