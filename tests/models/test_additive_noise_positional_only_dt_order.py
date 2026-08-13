"""Regression tests for positional-only additive-noise ``dt`` dispatch."""

from pyrecest.models import AdditiveNoiseTransitionModel


def test_positional_only_dt_after_defaulted_parameter_keeps_its_position():
    calls = []

    def transition(state, scale=2.0, dt=1.0, /):
        calls.append(("transition", state, scale, dt))
        return state + scale * dt

    def jacobian(_state, scale=2.0, dt=1.0, /):
        calls.append(("jacobian", scale, dt))
        return scale * dt

    model = AdditiveNoiseTransitionModel(
        transition,
        jacobian=jacobian,
        dt=0.5,
    )

    assert model.evaluate(1.0) == 2.0
    assert model.jacobian(1.0) == 1.0
    assert calls == [
        ("transition", 1.0, 2.0, 0.5),
        ("jacobian", 2.0, 0.5),
    ]


def test_positional_only_dt_preserves_configured_and_per_call_arguments():
    calls = []

    def transition(state, scale, dt, /):
        calls.append(("transition", state, scale, dt))
        return state + scale * dt

    def jacobian(_state, scale, dt, /):
        calls.append(("jacobian", scale, dt))
        return scale * dt

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
        ("transition", 1.0, 4.0, 0.5),
        ("jacobian", 4.0, 0.5),
        ("transition", 1.0, 6.0, 0.5),
        ("jacobian", 6.0, 0.5),
    ]


def test_var_positional_transition_still_receives_dt_positionally():
    def transition(state, *args):
        return state, args

    model = AdditiveNoiseTransitionModel(transition, dt=0.25)

    assert model.evaluate(1.0) == (1.0, (0.25,))
