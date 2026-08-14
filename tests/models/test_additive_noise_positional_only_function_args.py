"""Regression tests for positional-only additive-noise callback arguments."""

import pytest
from pyrecest.models import (
    AdditiveNoiseMeasurementModel,
    AdditiveNoiseTransitionModel,
)


def test_transition_callbacks_receive_positional_only_function_arguments():
    calls = []

    def transition(state, dt, scale, /):
        calls.append(("transition", state, dt, scale))
        return state + dt * scale

    def jacobian(state, dt, scale, /):
        calls.append(("jacobian", state, dt, scale))
        return dt * scale

    model = AdditiveNoiseTransitionModel(
        transition,
        jacobian=jacobian,
        dt=0.5,
        function_args={"scale": 4.0},
    )

    assert model.evaluate(1.0) == 3.0
    assert model.jacobian(1.0) == 2.0
    assert model.evaluate(1.0, scale=6.0) == 4.0
    assert model.jacobian(1.0, scale=6.0) == 3.0
    assert calls == [
        ("transition", 1.0, 0.5, 4.0),
        ("jacobian", 1.0, 0.5, 4.0),
        ("transition", 1.0, 0.5, 6.0),
        ("jacobian", 1.0, 0.5, 6.0),
    ]


def test_measurement_callbacks_preserve_positional_only_defaults():
    def measurement(state, scale=2.0, offset=1.0, /):
        return state * scale + offset

    def jacobian(_state, scale=2.0, offset=1.0, /):
        return scale + offset

    model = AdditiveNoiseMeasurementModel(
        measurement,
        jacobian=jacobian,
        function_args={"offset": 0.5},
    )

    assert model.evaluate(3.0) == 6.5
    assert model.jacobian(3.0) == 2.5
    assert model.evaluate(3.0, scale=4.0) == 12.5
    assert model.jacobian(3.0, scale=4.0) == 4.5


def test_model_callbacks_still_reject_unsupported_arguments():
    model = AdditiveNoiseMeasurementModel(
        lambda state: state,
        function_args={"unsupported": 1.0},
    )

    with pytest.raises(TypeError, match="unsupported"):
        model.evaluate(2.0)
