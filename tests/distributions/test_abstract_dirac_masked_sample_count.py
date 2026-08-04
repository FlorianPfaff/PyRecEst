"""Regression tests for masked Dirac sample counts."""

import numpy as np
import pytest

from pyrecest.backend import array
from pyrecest.distributions.se2_dirac_distribution import SE2DiracDistribution


def _single_particle_distribution() -> SE2DiracDistribution:
    return SE2DiracDistribution(array([[0.0, 0.0, 0.0]]))


def test_dirac_sampling_rejects_masked_sample_count():
    distribution = _single_particle_distribution()

    with pytest.raises(ValueError, match="positive integer"):
        distribution.sample(np.ma.array(3, mask=True))


def test_dirac_sampling_accepts_clear_mask_sample_count():
    distribution = _single_particle_distribution()

    samples = distribution.sample(np.ma.array(3, mask=False))

    assert samples.shape == (3, 3)
