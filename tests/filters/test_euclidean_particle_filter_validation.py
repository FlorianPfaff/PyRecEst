import numpy as np
import pytest
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter


@pytest.mark.parametrize(
    "n_particles",
    [np.int64(3), np.array(3, dtype=np.int64), np.array(3.0)],
)
@pytest.mark.parametrize(
    "dim",
    [np.int64(2), np.array(2, dtype=np.int64), np.array(2.0)],
)
def test_euclidean_particle_filter_accepts_integer_like_scalar_counts(n_particles, dim):
    particle_filter = EuclideanParticleFilter(n_particles=n_particles, dim=dim)

    assert particle_filter.filter_state.d.shape == (3, 2)


@pytest.mark.parametrize(
    "bad_value",
    [
        True,
        np.bool_(True),
        np.array(True),
        np.array([3]),
        1.5,
        np.array(1.5),
        "3",
        1 + 0j,
    ],
)
def test_euclidean_particle_filter_rejects_invalid_scalar_counts(bad_value):
    with pytest.raises(ValueError, match="n_particles must be a positive integer"):
        EuclideanParticleFilter(n_particles=bad_value, dim=1)

    with pytest.raises(ValueError, match="dim must be a positive integer"):
        EuclideanParticleFilter(n_particles=1, dim=bad_value)
