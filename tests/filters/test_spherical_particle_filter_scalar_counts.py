import numpy as np
import pytest
from pyrecest.filters.hyperhemispherical_particle_filter import (
    HyperhemisphericalParticleFilter,
)
from pyrecest.filters.hyperspherical_particle_filter import HypersphericalParticleFilter


@pytest.mark.parametrize(
    ("filter_type", "expected_width"),
    [
        (HypersphericalParticleFilter, 3),
        (HyperhemisphericalParticleFilter, 4),
    ],
)
def test_spherical_particle_filters_accept_scalar_array_counts(
    filter_type, expected_width
):
    particle_filter = filter_type(
        np.array(5, dtype=np.int64),
        np.ma.array(3, mask=False, dtype=np.int64),
    )

    assert particle_filter.filter_state.d.shape == (5, expected_width)


@pytest.mark.parametrize(
    "filter_type",
    [HypersphericalParticleFilter, HyperhemisphericalParticleFilter],
)
@pytest.mark.parametrize("argument", ["n_particles", "dim"])
def test_spherical_particle_filters_reject_masked_integer_controls(
    filter_type, argument
):
    arguments = {"n_particles": 5, "dim": 3}
    arguments[argument] = np.ma.array(3, mask=True, dtype=np.int64)

    with pytest.raises(ValueError, match=argument):
        filter_type(**arguments)


@pytest.mark.parametrize(
    "filter_type",
    [HypersphericalParticleFilter, HyperhemisphericalParticleFilter],
)
@pytest.mark.parametrize("argument", ["n_particles", "dim"])
def test_spherical_particle_filters_reject_boolean_scalar_arrays(
    filter_type, argument
):
    arguments = {"n_particles": 5, "dim": 3}
    arguments[argument] = np.array(True)

    with pytest.raises(ValueError, match=argument):
        filter_type(**arguments)
