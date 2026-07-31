import numpy as np
import pytest
from pyrecest._backend.numpy import random


@pytest.mark.parametrize(
    ("sampler", "message"),
    (
        (
            lambda: random.uniform(low=np.ma.array(0.0, mask=True)),
            "low must be real numeric",
        ),
        (
            lambda: random.uniform(high=np.ma.array(1.0, mask=True)),
            "high must be real numeric",
        ),
        (
            lambda: random.normal(loc=np.ma.array(0.0, mask=True)),
            "loc must be real numeric",
        ),
        (
            lambda: random.normal(scale=np.ma.array(1.0, mask=True)),
            "scale must be real numeric",
        ),
        (
            lambda: random.multivariate_normal(
                mean=np.ma.array([0.0, 1.0], mask=[False, True]),
                cov=np.eye(2),
            ),
            "mean must be real numeric",
        ),
        (
            lambda: random.multivariate_normal(
                mean=np.zeros(2),
                cov=np.ma.array(
                    np.eye(2),
                    mask=[[False, False], [False, True]],
                ),
            ),
            "cov must be real numeric",
        ),
        (
            lambda: random.choice(
                2,
                p=np.ma.array([0.25, 0.75], mask=[False, True]),
            ),
            "p must be real numeric",
        ),
        (
            lambda: random.choice(2, p=[0.25, np.ma.masked]),
            "p must be real numeric",
        ),
    ),
)
def test_numpy_random_rejects_masked_distribution_parameters(sampler, message):
    with pytest.raises(TypeError, match=message):
        sampler()


def test_numpy_random_accepts_fully_unmasked_distribution_parameters():
    uniform_samples = random.uniform(
        low=np.ma.array(0.0, mask=False),
        high=np.ma.array(1.0, mask=False),
        size=4,
    )
    normal_samples = random.normal(
        loc=np.ma.array(0.0, mask=False),
        scale=np.ma.array(1.0, mask=False),
        size=4,
    )
    multivariate_samples = random.multivariate_normal(
        mean=np.ma.array([0.0, 1.0], mask=False),
        cov=np.ma.array(np.eye(2), mask=False),
        size=3,
    )
    choice_samples = random.choice(
        2,
        size=4,
        p=np.ma.array([0.25, 0.75], mask=False),
    )

    assert uniform_samples.shape == (4,)
    assert normal_samples.shape == (4,)
    assert multivariate_samples.shape == (3, 2)
    assert choice_samples.shape == (4,)
    assert np.isfinite(uniform_samples).all()
    assert np.isfinite(normal_samples).all()
    assert np.isfinite(multivariate_samples).all()
    assert np.isin(choice_samples, [0, 1]).all()
