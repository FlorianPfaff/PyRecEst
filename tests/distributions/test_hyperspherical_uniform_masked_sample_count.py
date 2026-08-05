from __future__ import annotations

import numpy as np
import pytest
from pyrecest.distributions import (
    HyperhemisphericalUniformDistribution,
    HypersphericalUniformDistribution,
    SO3UniformDistribution,
)


@pytest.fixture(
    params=[
        HypersphericalUniformDistribution(2),
        HyperhemisphericalUniformDistribution(2),
        SO3UniformDistribution(),
    ],
    ids=["hyperspherical", "hyperhemispherical", "so3"],
)
def distribution(request):
    return request.param


def test_uniform_samplers_reject_masked_sample_counts(distribution) -> None:
    with pytest.raises(ValueError, match="integer"):
        distribution.sample(np.ma.array(4, mask=True))


def test_uniform_samplers_reject_object_wrapped_masked_sample_counts(
    distribution,
) -> None:
    masked_count = np.empty((), dtype=object)
    masked_count[()] = np.ma.masked

    with pytest.raises(ValueError, match="integer"):
        distribution.sample(masked_count)


def test_uniform_samplers_accept_clear_mask_sample_counts(distribution) -> None:
    samples = distribution.sample(np.ma.array(4, mask=False))

    assert samples.shape == (4, distribution.input_dim)
