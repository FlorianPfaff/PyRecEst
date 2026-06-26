import pytest
from pyrecest.models.likelihood import SampleableTransitionModel


def test_sampleable_transition_vectorization_flag_assignment_is_validated():
    def identity_state(state):
        return state

    model = SampleableTransitionModel(
        identity_state,
        function_is_vectorized=False,
    )

    assert model.function_is_vectorized is False
    model.function_is_vectorized = True
    assert model.function_is_vectorized is True

    for flag in ("False", 1, [True]):
        with pytest.raises(TypeError, match="function_is_vectorized"):
            SampleableTransitionModel(
                identity_state,
                function_is_vectorized=flag,
            )
        with pytest.raises(TypeError, match="function_is_vectorized"):
            model.function_is_vectorized = flag
