"""Validated wrappers for sensor-model array inputs."""

from __future__ import annotations

from typing import Any

from pyrecest.backend import is_complex

from . import sensor_models as _sensor_models

_state_vector_impl = _sensor_models._state_vector  # pylint: disable=protected-access
_as_vector_impl = _sensor_models._as_vector  # pylint: disable=protected-access
_as_matrix_impl = _sensor_models._as_matrix  # pylint: disable=protected-access
_as_sensor_positions_impl = (
    _sensor_models._as_sensor_positions  # pylint: disable=protected-access
)
_range_rate_impl = _sensor_models._range_rate  # pylint: disable=protected-access


def _require_real_array(value: Any, name: str):
    """Reject complex backend arrays used by real-valued sensor models."""
    if bool(is_complex(value)):
        raise ValueError(f"{name} must contain real values")
    return value


def _validated_state_vector(state: Any):
    """Return a real backend state vector after validating its rank."""
    try:
        state_vector = _state_vector_impl(state)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("state must be a one-dimensional array") from exc
    if len(tuple(state_vector.shape)) != 1:
        raise ValueError("state must be a one-dimensional array")
    return _require_real_array(state_vector, "state")


def _validated_vector(value: Any, length: int, name: str):
    """Return a real sensor-model vector with the requested shape."""
    return _require_real_array(_as_vector_impl(value, length, name), name)


def _validated_matrix(value: Any, shape: tuple[int, int], name: str):
    """Return a real sensor-model matrix with the requested shape."""
    return _require_real_array(_as_matrix_impl(value, shape, name), name)


def _validated_sensor_positions(value: Any, position_dim: int = 2):
    """Return real sensor positions after the existing shape validation."""
    return _require_real_array(
        _as_sensor_positions_impl(value, position_dim),
        "sensor_positions",
    )


def _validated_range_rate(position, velocity, sensor_position, sensor_velocity):
    """Reject complex FDOA sensor velocities before computing a range rate."""
    _require_real_array(sensor_velocity, "sensor_velocities")
    return _range_rate_impl(position, velocity, sensor_position, sensor_velocity)


def install_sensor_state_validation() -> None:
    """Install shared rank and real-array validation for sensor-model inputs."""
    _sensor_models._state_vector = (
        _validated_state_vector  # pylint: disable=protected-access
    )
    _sensor_models._as_vector = _validated_vector  # pylint: disable=protected-access
    _sensor_models._as_matrix = _validated_matrix  # pylint: disable=protected-access
    _sensor_models._as_sensor_positions = (
        _validated_sensor_positions  # pylint: disable=protected-access
    )
    _sensor_models._range_rate = (  # pylint: disable=protected-access
        _validated_range_rate
    )


__all__ = ["install_sensor_state_validation"]
