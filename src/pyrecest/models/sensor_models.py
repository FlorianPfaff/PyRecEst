"""Ready-made nonlinear sensor and measurement models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# pylint: disable=no-name-in-module,no-member,too-many-arguments,too-many-positional-arguments
from pyrecest.backend import arctan2, asarray, matvec, sqrt, stack
from pyrecest.backend import sum as _sum
from pyrecest.backend import zeros
from pyrecest.models.additive_noise import AdditiveNoiseMeasurementModel

__all__ = [
    "bearing_only_measurement",
    "bearing_only_model",
    "camera_projection_measurement",
    "camera_projection_model",
    "fdoa_measurement",
    "fdoa_model",
    "radar_range_bearing_doppler_measurement",
    "radar_range_bearing_doppler_model",
    "range_bearing_jacobian",
    "range_bearing_measurement",
    "range_bearing_model",
    "tdoa_measurement",
    "tdoa_model",
]

_TEXT_OR_BOOL_SCALAR_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
)
_REJECTED_SCALAR_ARRAY_KINDS = frozenset({"b", "c", "S", "U", "M", "m"})


def _state_vector(state):
    return asarray(state)


def _as_scalar_float(value: Any, name: str) -> float:
    value_array = np.asarray(value)
    if (
        value_array.shape != ()
        or value_array.dtype.kind in _REJECTED_SCALAR_ARRAY_KINDS
    ):
        raise ValueError(f"{name} must be a scalar number")
    scalar_value = value_array.item()
    if isinstance(scalar_value, _TEXT_OR_BOOL_SCALAR_TYPES) or isinstance(
        scalar_value, (complex, np.complexfloating)
    ):
        raise ValueError(f"{name} must be a scalar number")
    try:
        scalar = float(scalar_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a scalar number") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_positive_float(value: Any, name: str) -> float:
    scalar = _as_scalar_float(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _as_integer(value: Any, name: str) -> int:
    scalar = _as_scalar_float(value, name)
    if not scalar.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(scalar)


def _normalize_indices(
    indices: Sequence[int],
    state_dim: int,
    expected_length: int,
    name: str,
) -> tuple[int, ...]:
    try:
        values = tuple(indices)
    except TypeError as exc:
        raise ValueError(f"{name} must contain {expected_length} indices") from exc
    if len(values) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} indices")
    normalized = tuple(
        _as_integer(index, f"{name}[{axis}]") for axis, index in enumerate(values)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} entries must be distinct")
    if any(index < 0 or index >= state_dim for index in normalized):
        raise ValueError(f"{name} entries must be valid state indices")
    return normalized


def _select(state, indices: Sequence[int], expected_length: int, name: str):
    state = _state_vector(state)
    normalized_indices = _normalize_indices(
        indices,
        int(state.shape[0]),
        expected_length,
        name,
    )
    return stack([state[index] for index in normalized_indices])


def _normalize_reference_sensor(reference_sensor: int, sensor_count: int) -> int:
    reference_sensor = _as_integer(reference_sensor, "reference_sensor")
    if reference_sensor < 0 or reference_sensor >= sensor_count:
        raise ValueError("reference_sensor must be a valid sensor index")
    return reference_sensor


def _as_vector(value, length: int, name: str):
    if value is None:
        return zeros((length,))
    vector = asarray(value)
    if tuple(vector.shape) != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    return vector


def _as_sensor_positions(value, position_dim: int = 2):
    sensors = asarray(value)
    sensor_shape = tuple(sensors.shape)
    if len(sensor_shape) != 2 or sensor_shape[1] != position_dim:
        raise ValueError(
            f"sensor_positions must have shape (n_sensors, {position_dim})"
        )
    if sensor_shape[0] < 2:
        raise ValueError("sensor_positions must contain at least two sensors")
    return sensors


def _validate_positive_range_squared(range_sq, name="range"):
    range_sq_value = _as_scalar_float(range_sq, name)
    if range_sq_value <= 0.0:
        raise ValueError(f"{name} is undefined for zero distance to the sensor")
    return range_sq


def _validate_nonzero_scalar(value, name: str):
    scalar = _as_scalar_float(value, name)
    if scalar == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return value


def range_bearing_measurement(
    state, sensor_position: Any | None = None, position_indices: Sequence[int] = (0, 1)
):
    """Return 2D range-bearing measurement ``[range, bearing]``."""
    position = _select(state, position_indices, 2, "position_indices")
    sensor_position = _as_vector(sensor_position, 2, "sensor_position")
    relative = position - sensor_position
    range_sq = relative[0] * relative[0] + relative[1] * relative[1]
    _validate_positive_range_squared(range_sq)
    range_value = sqrt(range_sq)
    bearing = arctan2(relative[1], relative[0])
    return stack([range_value, bearing])


def bearing_only_measurement(
    state, sensor_position: Any | None = None, position_indices: Sequence[int] = (0, 1)
):
    """Return a scalar 2D bearing-only measurement."""
    return stack(
        [
            range_bearing_measurement(
                state,
                sensor_position=sensor_position,
                position_indices=position_indices,
            )[1]
        ]
    )


def range_bearing_jacobian(
    state, sensor_position: Any | None = None, position_indices: Sequence[int] = (0, 1)
):
    """Return the Jacobian of :func:`range_bearing_measurement`.

    The Jacobian has shape ``(2, state_dim)`` and nonzero columns only at
    ``position_indices``.
    """
    state = _state_vector(state)
    position_indices = _normalize_indices(
        position_indices,
        int(state.shape[0]),
        2,
        "position_indices",
    )
    position = stack([state[index] for index in position_indices])
    sensor_position = _as_vector(sensor_position, 2, "sensor_position")
    dx = position[0] - sensor_position[0]
    dy = position[1] - sensor_position[1]
    range_sq = dx * dx + dy * dy
    _validate_positive_range_squared(range_sq, "range_bearing_jacobian")
    range_value = sqrt(range_sq)
    state_dim = int(state.shape[0])
    x_index = position_indices[0]
    y_index = position_indices[1]
    zero = range_value * 0.0
    range_entries = []
    bearing_entries = []
    for column in range(state_dim):
        if column == x_index:
            range_entries.append(dx / range_value)
            bearing_entries.append(-dy / range_sq)
        elif column == y_index:
            range_entries.append(dy / range_value)
            bearing_entries.append(dx / range_sq)
        else:
            range_entries.append(zero)
            bearing_entries.append(zero)
    return stack([stack(range_entries), stack(bearing_entries)])


def radar_range_bearing_doppler_measurement(
    state,
    sensor_position: Any | None = None,
    sensor_velocity: Any | None = None,
    position_indices: Sequence[int] = (0, 1),
    velocity_indices: Sequence[int] = (2, 3),
):
    """Return 2D radar ``[range, bearing, range_rate]`` measurement."""
    position = _select(state, position_indices, 2, "position_indices")
    velocity = _select(state, velocity_indices, 2, "velocity_indices")
    sensor_position = _as_vector(sensor_position, 2, "sensor_position")
    sensor_velocity = _as_vector(sensor_velocity, 2, "sensor_velocity")
    relative_position = position - sensor_position
    relative_velocity = velocity - sensor_velocity
    range_sq = (
        relative_position[0] * relative_position[0]
        + relative_position[1] * relative_position[1]
    )
    _validate_positive_range_squared(range_sq, "radar range")
    range_value = sqrt(range_sq)
    bearing = arctan2(relative_position[1], relative_position[0])
    range_rate = (
        relative_position[0] * relative_velocity[0]
        + relative_position[1] * relative_velocity[1]
    ) / range_value
    return stack([range_value, bearing, range_rate])


def tdoa_measurement(
    state,
    sensor_positions: Any,
    reference_sensor: int = 0,
    propagation_speed: float = 1.0,
    position_indices: Sequence[int] = (0, 1),
):
    """Return TDOA range-difference measurements relative to one sensor."""
    propagation_speed = _as_positive_float(propagation_speed, "propagation_speed")
    position = _select(state, position_indices, 2, "position_indices")
    sensors = _as_sensor_positions(sensor_positions)
    reference_sensor = _normalize_reference_sensor(
        reference_sensor,
        int(sensors.shape[0]),
    )
    reference_relative = position - sensors[reference_sensor]
    reference_range = sqrt(_sum(reference_relative * reference_relative))
    measurements = []
    for sensor_idx in range(int(sensors.shape[0])):
        if sensor_idx == reference_sensor:
            continue
        relative = position - sensors[sensor_idx]
        sensor_range = sqrt(_sum(relative * relative))
        measurements.append((sensor_range - reference_range) / propagation_speed)
    return stack(measurements)


def fdoa_measurement(
    state,
    sensor_positions: Any,
    sensor_velocities: Any | None = None,
    reference_sensor: int = 0,
    propagation_speed: float = 1.0,
    position_indices: Sequence[int] = (0, 1),
    velocity_indices: Sequence[int] = (2, 3),
):
    """Return FDOA range-rate-difference measurements relative to one sensor."""
    propagation_speed = _as_positive_float(propagation_speed, "propagation_speed")
    position = _select(state, position_indices, 2, "position_indices")
    velocity = _select(state, velocity_indices, 2, "velocity_indices")
    sensors = _as_sensor_positions(sensor_positions)
    if sensor_velocities is None:
        sensor_velocities = zeros(tuple(sensors.shape))
    sensor_velocities = asarray(sensor_velocities)
    if tuple(sensor_velocities.shape) != tuple(sensors.shape):
        raise ValueError(
            "sensor_velocities must have the same shape as sensor_positions"
        )
    reference_sensor = _normalize_reference_sensor(
        reference_sensor,
        int(sensors.shape[0]),
    )
    reference_rate = _range_rate(
        position,
        velocity,
        sensors[reference_sensor],
        sensor_velocities[reference_sensor],
    )
    measurements = []
    for sensor_idx in range(int(sensors.shape[0])):
        if sensor_idx == reference_sensor:
            continue
        rate = _range_rate(
            position, velocity, sensors[sensor_idx], sensor_velocities[sensor_idx]
        )
        measurements.append((rate - reference_rate) / propagation_speed)
    return stack(measurements)


def camera_projection_measurement(
    state,
    camera_matrix: Any | None = None,
    rotation: Any | None = None,
    translation: Any | None = None,
    position_indices: Sequence[int] = (0, 1, 2),
):
    """Return pinhole camera image coordinates for a 3D point.

    ``rotation`` and ``translation`` transform world coordinates into camera
    coordinates using ``p_camera = rotation @ p_world + translation``. If
    ``camera_matrix`` is omitted, normalized image coordinates ``[x/z, y/z]``
    are returned.
    """
    position = _select(state, position_indices, 3, "position_indices")
    rotation = asarray(rotation) if rotation is not None else None
    translation = _as_vector(translation, 3, "translation")
    camera_position = position if rotation is None else matvec(rotation, position)
    camera_position = camera_position + translation
    _validate_nonzero_scalar(camera_position[2], "camera depth")
    normalized = stack(
        [
            camera_position[0] / camera_position[2],
            camera_position[1] / camera_position[2],
            camera_position[2] / camera_position[2],
        ]
    )
    if camera_matrix is None:
        return stack([normalized[0], normalized[1]])
    homogeneous = matvec(asarray(camera_matrix), normalized)
    _validate_nonzero_scalar(homogeneous[2], "homogeneous camera scale")
    return stack([homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]])


def range_bearing_model(
    noise_covariance,
    sensor_position: Any | None = None,
    position_indices: Sequence[int] = (0, 1),
) -> AdditiveNoiseMeasurementModel:
    """Return an additive-noise 2D range-bearing model."""
    return AdditiveNoiseMeasurementModel(
        measurement_function=range_bearing_measurement,
        noise_covariance=noise_covariance,
        jacobian=lambda state: range_bearing_jacobian(
            state, sensor_position=sensor_position, position_indices=position_indices
        ),
        function_args={
            "sensor_position": sensor_position,
            "position_indices": tuple(position_indices),
        },
    )


def bearing_only_model(
    noise_covariance,
    sensor_position: Any | None = None,
    position_indices: Sequence[int] = (0, 1),
) -> AdditiveNoiseMeasurementModel:
    """Return an additive-noise 2D bearing-only model."""
    return AdditiveNoiseMeasurementModel(
        measurement_function=bearing_only_measurement,
        noise_covariance=noise_covariance,
        function_args={
            "sensor_position": sensor_position,
            "position_indices": tuple(position_indices),
        },
    )


def radar_range_bearing_doppler_model(
    noise_covariance,
    sensor_position: Any | None = None,
    sensor_velocity: Any | None = None,
    position_indices: Sequence[int] = (0, 1),
    velocity_indices: Sequence[int] = (2, 3),
) -> AdditiveNoiseMeasurementModel:
    """Return an additive-noise radar range-bearing-Doppler model."""
    return AdditiveNoiseMeasurementModel(
        measurement_function=radar_range_bearing_doppler_measurement,
        noise_covariance=noise_covariance,
        function_args={
            "sensor_position": sensor_position,
            "sensor_velocity": sensor_velocity,
            "position_indices": tuple(position_indices),
            "velocity_indices": tuple(velocity_indices),
        },
    )


def tdoa_model(
    noise_covariance,
    sensor_positions: Any,
    reference_sensor: int = 0,
    propagation_speed: float = 1.0,
    position_indices: Sequence[int] = (0, 1),
) -> AdditiveNoiseMeasurementModel:
    """Return an additive-noise TDOA model."""
    return AdditiveNoiseMeasurementModel(
        measurement_function=tdoa_measurement,
        noise_covariance=noise_covariance,
        function_args={
            "sensor_positions": sensor_positions,
            "reference_sensor": reference_sensor,
            "propagation_speed": propagation_speed,
            "position_indices": tuple(position_indices),
        },
    )


def fdoa_model(
    noise_covariance,
    sensor_positions: Any,
    sensor_velocities: Any | None = None,
    reference_sensor: int = 0,
    propagation_speed: float = 1.0,
    position_indices: Sequence[int] = (0, 1),
    velocity_indices: Sequence[int] = (2, 3),
) -> AdditiveNoiseMeasurementModel:
    """Return an additive-noise FDOA model."""
    return AdditiveNoiseMeasurementModel(
        measurement_function=fdoa_measurement,
        noise_covariance=noise_covariance,
        function_args={
            "sensor_positions": sensor_positions,
            "sensor_velocities": sensor_velocities,
            "reference_sensor": reference_sensor,
            "propagation_speed": propagation_speed,
            "position_indices": tuple(position_indices),
            "velocity_indices": tuple(velocity_indices),
        },
    )


def camera_projection_model(
    noise_covariance,
    camera_matrix: Any | None = None,
    rotation: Any | None = None,
    translation: Any | None = None,
    position_indices: Sequence[int] = (0, 1, 2),
) -> AdditiveNoiseMeasurementModel:
    """Return an additive-noise pinhole camera projection model."""
    return AdditiveNoiseMeasurementModel(
        measurement_function=camera_projection_measurement,
        noise_covariance=noise_covariance,
        function_args={
            "camera_matrix": camera_matrix,
            "rotation": rotation,
            "translation": translation,
            "position_indices": tuple(position_indices),
        },
    )


def _range_rate(position, velocity, sensor_position, sensor_velocity):
    relative_position = position - sensor_position
    relative_velocity = velocity - sensor_velocity
    range_sq = _sum(relative_position * relative_position)
    _validate_positive_range_squared(range_sq, "range rate")
    range_value = sqrt(range_sq)
    return _sum(relative_position * relative_velocity) / range_value
