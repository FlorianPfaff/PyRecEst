"""Transition-callback calling contract for the unscented RTS smoother."""

from __future__ import annotations

import inspect
from typing import Callable

from pyrecest.backend import asarray

from . import unscented_rauch_tung_striebel_smoother as _urts

# pylint: disable=protected-access


def _time_step_call_mode(function: Callable) -> tuple[str, str | None] | None:
    """Return how a transition callback accepts its time-step argument."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return None

    parameters = tuple(signature.parameters.values())
    for parameter_name in ("dt", "time_step"):
        time_parameter = signature.parameters.get(parameter_name)
        if time_parameter is None:
            continue
        if time_parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            return "positional", None
        return "keyword", parameter_name

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return "keyword", "dt"

    positional_parameters = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if (
        any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        or len(positional_parameters) >= 2
    ):
        return "positional", None
    return None


def _call_transition(function: Callable, sigma_point, time_step):
    """Call a transition callback without dropping supported time-step arguments."""

    if time_step is None:
        return asarray(function(sigma_point)).reshape(-1)

    call_mode = _time_step_call_mode(function)
    if call_mode is None:
        return asarray(function(sigma_point)).reshape(-1)

    mode, parameter_name = call_mode
    if mode == "positional":
        return asarray(function(sigma_point, time_step)).reshape(-1)
    return asarray(function(sigma_point, **{parameter_name: time_step})).reshape(-1)


def install_urts_transition_call_contract() -> None:
    """Install keyword-aware transition dispatch on the unscented RTS smoother."""

    marker = "_pyrecest_time_step_call_contract"
    setattr(_call_transition, marker, True)
    current = _urts.UnscentedRauchTungStriebelSmoother._call_transition
    if not getattr(current, marker, False):
        _urts.UnscentedRauchTungStriebelSmoother._call_transition = staticmethod(
            _call_transition
        )
