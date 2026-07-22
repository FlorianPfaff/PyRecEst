"""Validated wrappers for sensor-model state inputs."""

from __future__ import annotations

from typing import Any

from . import sensor_models as _sensor_models

_state_vector_impl = _sensor_models._state_vector  # pylint: disable=protected-access


def _validated_state_vector(state: Any):
    """Return a backend state vector after validating its rank."""
    try:
        state_vector = _state_vector_impl(state)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("state must be a one-dimensional array") from exc
    if len(tuple(state_vector.shape)) != 1:
        raise ValueError("state must be a one-dimensional array")
    return state_vector


def install_sensor_state_validation() -> None:
    """Install shared rank validation for sensor-model state inputs."""
    _sensor_models._state_vector = (
        _validated_state_vector  # pylint: disable=protected-access
    )


__all__ = ["install_sensor_state_validation"]
