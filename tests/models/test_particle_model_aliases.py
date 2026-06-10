"""Regression tests for legacy particle model import aliases."""

from pyrecest.backend import allclose, arange, array, reshape
from pyrecest.models import (
    LikelihoodMeasurementModel,
    SampleableTransitionModel,
)
from pyrecest.models.particle import (
    LikelihoodMeasurementModel as ParticleLikelihoodMeasurementModel,
)
from pyrecest.models.particle import (
    SampleableTransitionModel as ParticleSampleableTransitionModel,
)


def test_particle_model_imports_share_canonical_implementations():
    assert ParticleLikelihoodMeasurementModel is LikelihoodMeasurementModel
    assert ParticleSampleableTransitionModel is SampleableTransitionModel


def test_particle_sampleable_alias_supports_public_constructor_features():
    def sample_next(state, *, n=1):
        return state + reshape(arange(n), (n, 1))

    model = ParticleSampleableTransitionModel(
        sample_next,
        transition_density=lambda state_next, state_previous: state_next
        - state_previous,
        name="legacy-particle-transition",
        function_is_vectorized=False,
    )

    assert model.name == "legacy-particle-transition"
    assert model.function_is_vectorized is False
    assert model.has_transition_density
    assert allclose(model.sample_next(array([4.0]), n=2), array([[4.0], [5.0]]))
    assert allclose(model.transition_density(array([3.0]), array([1.0])), array([2.0]))
