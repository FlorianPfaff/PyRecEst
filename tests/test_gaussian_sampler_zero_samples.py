import numpy as np
from pyrecest.sampling.euclidean_sampler import GaussianSampler


def test_gaussian_sampler_zero_samples_returns_empty_matrix():
    samples = GaussianSampler().sample_stochastic(0, 3)

    assert samples.shape == (0, 3)
    assert np.isfinite(np.asarray(samples)).all()
