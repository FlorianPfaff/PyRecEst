"""Ready-made nonlinear sensor and measurement models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# pylint: disable=no-name-in-module,no-member,too-many-arguments,too-many-positional-arguments
from pyrecest.backend import arctan2, asarray, matmul, sqrt, stack
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


def _state_vector(state):
    return asarray(state)


def _select(state, indices: Sequence[int]):
    state = _state_vector(state)
    return stack([state[int(index)] for index in indices])


def _as_vector(value, length: int, name: str):
    if value is None:
        return zeros((length,))
    vector = asarray(value)
    if tuple(vector.shape) != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    return vector


def range_bearing_measurement(
    state, sensor_position: Any | None = None, position_indices: Sequence[int] = (0, 1)
):
    """Return 2D range-bearing measurement ``[range, bearing]``."""
    position = _select(state, position_indices)
    sensor_position = _as_vector(sensor_position, 2, "sensor_position")
    relative = position - sensor_position
    range_value = sqrt(relative[0] * relative[0] + relative[1] * relative[1])
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
    position = _select(state, position_indices)
    sensor_position = _as_vector(sensor_position, 2, "sensor_position")
    dx = position[0] - sensor_position[0]
    dy = position[1] - sensor_position[1]
    range_sq = dx * dx + dy * dy
    range_value = sqrt(range_sq)
    state_dim = int(state.shape[0])
    range_row = [0.0] * state_dim
    bearing_row = [0.0] * state_dim
    range_row[int(position_indices[0])] = dx / range_value
    range_row[int(position_indices[1])] = dy / range_value
    bearing_row[int(position_indices[0])] = -dy / range_sq
    bearing_row[int(position_indices[1])] = dx / range_sq
    return stack([stack(range_row), stack(bearing_row)])


def radar_range_bearing_doppler_measurement(
    state,
    sensor_position: Any | None = None,
    sensor_velocity: Any | None = None,
    position_indices: Sequence[int] = (0, 1),
    velocity_indices: Sequence[int] = (2, 3),
):
    """Return 2D radar ``[range, bearing, range_rate]`` measurement."""
    position = _select(state, position_indices)
    velocity = _select(state, velocity_indices)
    sensor_position = _as_vector(sensor_position, 2, "sensor_position")
    sensor_velocity = _as_vector(sensor_velocity, 2, "sensor_velocity")
    relative_position = position - sensor_position
    relative_velocity = velocity - sensor_velocity
    range_value = sqrt(
        relative_position[0] * relative_position[0]
        + relative_position[1] * relative_position[1]
    )
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
    if float(propagation_speed) <= 0.0:
        raise ValueError("propagation_speed must be positive")
    position = _select(state, position_indices)
    sensors = asarray(sensor_positions)
    reference_sensor = int(reference_sensor)
    reference_relative = position - sensors[reference_sensor]
    reference_range = sqrt(_sum(reference_relative * reference_relative))
    measurements = []
    for sensor_idx in range(int(sensors.shape[0])):
        if sensor_idx == reference_sensor:
            continue
        relative = position - sensors[sensor_idx]
        sensor_range = sqrt(_sum(relative * relative))
        measurements.append((sensor_range - reference_range) / float(propagation_speed))
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
    if float(propagation_speed) <= 0.0:
        raise ValueError("propagation_speed must be positive")
    position = _select(state, position_indices)
    velocity = _select(state, velocity_indices)
    sensors = asarray(sensor_positions)
    if sensor_velocities is None:
        sensor_velocities = zeros(tuple(sensors.shape))
    sensor_velocities = asarray(sensor_velocities)
    reference_sensor = int(reference_sensor)
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
        measurements.append((rate - reference_rate) / float(propagation_speed))
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
    position = _select(state, position_indices)
    rotation = asarray(rotation) if rotation is not None else None
    translation = _as_vector(translation, 3, "translation")
    camera_position = position if rotation is None else matmul(rotation, position)
    camera_position = camera_position + translation
    normalized = stack(
        [
            camera_position[0] / camera_position[2],
            camera_position[1] / camera_position[2],
            1.0,
        ]
    )
    if camera_matrix is None:
        return stack([normalized[0], normalized[1]])
    homogeneous = matmul(asarray(camera_matrix), normalized)
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
    range_value = sqrt(_sum(relative_position * relative_position))
    return _sum(relative_position * relative_velocity) / range_value
