import numpy as np
import pytest

from pyrecest.sampling.euclidean_sampler import (
    FibonacciGridSampler,
    FibonacciRejectionSampler,
    GaussianSampler,
    HaltonGridSampler,
    SobolGridSampler,
)


@pytest.mark.parametrize(
    "sampler",
    [
        GaussianSampler(),
        FibonacciGridSampler(),
        SobolGridSampler(),
        HaltonGridSampler(),
    ],
)
def test_euclidean_samplers_reject_masked_count_and_dimension(sampler):
    with pytest.raises(ValueError):
        sampler.sample_stochastic(np.ma.array(4, mask=True), 2)
    with pytest.raises(ValueError):
        sampler.sample_stochastic(4, np.ma.array(2, mask=True))


def test_fibonacci_rejection_rejects_masked_scalar_controls():
    sampler = FibonacciRejectionSampler()

    def density(xs):
        return np.ones(xs.shape[0])

    with pytest.raises(ValueError, match="n_candidates"):
        sampler.sample_rejection(
            density,
            n_candidates=np.ma.array(4, mask=True),
            dim=2,
            max_density=1.0,
        )
    with pytest.raises(ValueError, match="dim"):
        sampler.sample_rejection(
            density,
            n_candidates=4,
            dim=np.ma.array(2, mask=True),
            max_density=1.0,
        )
    with pytest.raises(ValueError, match="max_density"):
        sampler.sample_rejection(
            density,
            n_candidates=4,
            dim=2,
            max_density=np.ma.array(1.0, mask=True),
        )


def test_fibonacci_rejection_accepts_clear_mask_scalar_wrappers():
    samples, info = FibonacciRejectionSampler().sample_rejection(
        lambda xs: np.ones(xs.shape[0]),
        n_candidates=np.ma.array(4, mask=False),
        dim=np.ma.array(2, mask=False),
        max_density=np.ma.array(1.0, mask=False),
    )

    assert samples.shape == (4, 2)
    assert info["n_candidates"] == 4
