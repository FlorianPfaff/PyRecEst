import copy

import numpy.testing as npt
import pytest
from pyrecest.filters.hyperspherical_particle_filter import (
    HypersphericalParticleFilter,
)


@pytest.mark.parametrize("function_is_vectorized", [True, False])
def test_predict_nonlinear_rejects_shape_change_atomically(function_is_vectorized):
    particle_filter = HypersphericalParticleFilter(n_particles=4, dim=3)
    original_state = particle_filter.filter_state
    original_particles = copy.deepcopy(original_state.d)
    original_weights = copy.deepcopy(original_state.w)

    def transition(values):
        return values[..., :-1]

    with pytest.raises(ValueError, match="returned particles with shape"):
        particle_filter.predict_nonlinear(
            transition,
            noise_distribution=None,
            function_is_vectorized=function_is_vectorized,
        )

    assert particle_filter.filter_state is original_state
    npt.assert_allclose(particle_filter.filter_state.d, original_particles)
    npt.assert_allclose(particle_filter.filter_state.w, original_weights)
