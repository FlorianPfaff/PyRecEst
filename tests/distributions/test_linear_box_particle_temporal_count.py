import numpy as np
import pytest
from pyrecest.backend import array
from pyrecest.distributions.nonperiodic.linear_box_particle_distribution import (
    LinearBoxParticleDistribution,
)


@pytest.mark.parametrize(
    "particle_count",
    [
        np.timedelta64(4, "ns"),
        np.timedelta64(4, "us"),
    ],
)
def test_sample_rejects_temporal_particle_counts(particle_count):
    distribution = LinearBoxParticleDistribution(array([[0.0]]), array([[1.0]]))

    with pytest.raises(ValueError, match="positive integer"):
        distribution.sample(particle_count)


@pytest.mark.parametrize(
    "particle_count",
    [
        np.timedelta64(4, "ns"),
        np.timedelta64(4, "us"),
    ],
)
def test_from_distribution_rejects_temporal_particle_counts(particle_count):
    with pytest.raises(ValueError, match="positive integer"):
        LinearBoxParticleDistribution.from_distribution(
            object(), n_particles=particle_count
        )
