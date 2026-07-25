import numpy as np
import pytest
from pyrecest.sampling.euclidean_sampler import FibonacciRejectionSampler


@pytest.mark.parametrize(
    "density_result",
    [
        np.asarray(0.5 + 0.25j),
        np.full(8, 0.5 + 0.25j),
    ],
)
def test_rejection_sampler_rejects_complex_density_values(density_result):
    sampler = FibonacciRejectionSampler()

    with pytest.raises(ValueError, match="pdf must return real density values"):
        sampler.sample_rejection(
            lambda _samples: density_result,
            n_candidates=8,
            dim=1,
            max_density=1.0,
        )
