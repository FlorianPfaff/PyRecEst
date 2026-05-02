"""Tests for reusable model object base protocols."""

from __future__ import annotations

import pytest

from pyrecest.models import (
    HasLikelihood,
    HasLinearMeasurement,
    HasLinearTransition,
    HasMeasurementFunction,
    HasTransitionDensity,
    HasTransitionFunction,
    HasTransitionSampler,
    MeasurementModel,
    Model,
    ModelMetadata,
    TransitionModel,
    require_capability,
    validate_optional_dimension,
)


class DummyLinearTransition:
    state_dim = 2
    system_matrix = ((1.0, 0.0), (0.0, 1.0))
    system_noise_cov = ((0.1, 0.0), (0.0, 0.1))
    sys_input = None


class DummyLinearMeasurement:
    state_dim = 2
    measurement_dim = 1
    measurement_matrix = ((1.0, 0.0),)
    meas_noise = ((0.5,),)


class DummyNonlinearModel:
    state_dim = 2
    measurement_dim = 1

    def transition_function(self, state):
        return state

    def measurement_function(self, state):
        return state[0]


class DummySamplingAndDensityModel:
    state_dim = 2

    def sample_next(self, state, num_samples=1):
        if num_samples == 1:
            return state
        return tuple(state for _ in range(num_samples))

    def transition_density(self, next_state, previous_state):
        return 1.0 if next_state == previous_state else 0.0

    def likelihood(self, measurement, state):
        return 1.0 if measurement == state[0] else 0.0


def test_model_metadata_accepts_optional_dimensions():
    metadata = ModelMetadata(name="demo", state_dim=2, measurement_dim=1)

    assert metadata.name == "demo"
    assert metadata.state_dim == 2
    assert metadata.measurement_dim == 1


def test_model_metadata_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        ModelMetadata(state_dim=0)

    with pytest.raises(TypeError):
        ModelMetadata(measurement_dim=1.5)  # type: ignore[arg-type]


def test_marker_classes_expose_metadata_properties():
    transition_model = TransitionModel()
    transition_model.metadata = ModelMetadata(name="transition", state_dim=2)

    measurement_model = MeasurementModel()
    measurement_model.metadata = ModelMetadata(name="measurement", state_dim=2, measurement_dim=1)

    assert isinstance(transition_model, Model)
    assert transition_model.name == "transition"
    assert transition_model.state_dim == 2
    assert isinstance(measurement_model, Model)
    assert measurement_model.measurement_dim == 1


def test_linear_transition_protocol_is_runtime_checkable():
    model = DummyLinearTransition()

    assert isinstance(model, HasLinearTransition)
    assert require_capability(model, HasLinearTransition) is model


def test_linear_measurement_protocol_is_runtime_checkable():
    model = DummyLinearMeasurement()

    assert isinstance(model, HasLinearMeasurement)
    assert require_capability(model, HasLinearMeasurement) is model


def test_function_protocols_are_runtime_checkable():
    model = DummyNonlinearModel()

    assert isinstance(model, HasTransitionFunction)
    assert isinstance(model, HasMeasurementFunction)


def test_sampling_density_and_likelihood_protocols_are_runtime_checkable():
    model = DummySamplingAndDensityModel()

    assert isinstance(model, HasTransitionSampler)
    assert isinstance(model, HasTransitionDensity)
    assert isinstance(model, HasLikelihood)


def test_require_capability_rejects_missing_capability():
    with pytest.raises(TypeError, match="HasLinearMeasurement"):
        require_capability(DummyLinearTransition(), HasLinearMeasurement)


def test_validate_optional_dimension_accepts_none_or_positive_int():
    assert validate_optional_dimension("state_dim", None) is None
    assert validate_optional_dimension("state_dim", 3) == 3
