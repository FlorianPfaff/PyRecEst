import numpy as np
import pytest
from pyrecest.sampling.euclidean_sampler import (
    FibonacciGridSampler,
    FibonacciRejectionSampler,
    HaltonGridSampler,
    SobolGridSampler,
)


@pytest.mark.parametrize("sampler", [SobolGridSampler(), HaltonGridSampler()])
def test_qmc_grid_samplers_reject_non_integral_grid_arguments(sampler):
    with pytest.raises(ValueError, match="n_samples"):
        sampler.get_uniform_samples(2.5, 2)
    with pytest.raises(ValueError, match="dim"):
        sampler.get_uniform_samples(4, 1.5)


@pytest.mark.parametrize("sampler", [SobolGridSampler(), HaltonGridSampler()])
def test_qmc_grid_samplers_accept_integer_like_scalar_arguments(sampler):
    samples = sampler.get_uniform_samples(np.array(4), np.float64(2.0))
    assert samples.shape == (4, 2)


def test_fibonacci_grid_sampler_rejects_non_integral_grid_arguments():
    sampler = FibonacciGridSampler()

    with pytest.raises(ValueError, match="n_points"):
        sampler.get_uniform_samples(2.5, 2)
    with pytest.raises(ValueError, match="d"):
        sampler.get_uniform_samples(4, 1.5)


def test_fibonacci_grid_sampler_accepts_integer_like_scalar_arguments():
    samples = FibonacciGridSampler().get_uniform_samples(np.float64(4.0), np.array(2))
    assert samples.shape == (4, 2)


def test_fibonacci_rejection_sampler_rejects_non_integral_rejection_arguments():
    sampler = FibonacciRejectionSampler()

    with pytest.raises(ValueError, match="n_candidates"):
        sampler.sample_rejection(
            lambda xs: np.ones(xs.shape[0]),
            n_candidates=2.5,
            dim=2,
            max_density=1.0,
        )
    with pytest.raises(ValueError, match="dim"):
        sampler.sample_rejection(
            lambda xs: np.ones(xs.shape[0]),
            n_candidates=4,
            dim=1.5,
            max_density=1.0,
        )


def test_fibonacci_rejection_sampler_accepts_integer_like_scalar_arguments():
    samples, info = FibonacciRejectionSampler().sample_rejection(
        lambda xs: np.ones(xs.shape[0]),
        n_candidates=np.float64(4.0),
        dim=np.array(2),
        max_density=1.0,
    )

    assert samples.shape == (4, 2)
    assert info["n_candidates"] == 4
