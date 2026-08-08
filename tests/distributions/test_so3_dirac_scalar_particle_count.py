"""Regression tests for SO(3) Dirac particle-count scalar handling."""

import numpy as np
import pytest

from pyrecest.distributions.so3_dirac_distribution import SO3DiracDistribution


class _SamplingDistribution:
    def __init__(self):
        self.sample_counts = []

    def sample(self, n):
        self.sample_counts.append(n)
        return np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))


@pytest.mark.parametrize(
    "count",
    [
        np.array(3, dtype=np.int64),
        np.ma.array(3, mask=False),
    ],
)
def test_from_distribution_accepts_integer_scalar_arrays(count):
    source = _SamplingDistribution()

    result = SO3DiracDistribution.from_distribution(source, n_particles=count)

    assert source.sample_counts == [3]
    assert result.d.shape == (3, 4)


@pytest.mark.parametrize(
    "count",
    [
        np.array(3.0),
        np.array([3]),
        np.ma.array(3, mask=True),
        True,
        np.bool_(True),
    ],
)
def test_from_distribution_rejects_non_integer_or_masked_scalars(count):
    with pytest.raises(ValueError, match="positive integer"):
        SO3DiracDistribution.from_distribution(
            _SamplingDistribution(),
            n_particles=count,
        )
