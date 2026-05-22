"""Vectorized normal-flow helpers for DVS contour scoring."""

from __future__ import annotations

import numpy as np

# pylint: disable=no-name-in-module
from pyrecest.backend import array

from .trackers import DVSFullSCGPTracker


def tracker_signed_normal_flows_vectorized(
    tracker: DVSFullSCGPTracker,
    event_xy: np.ndarray,
    event_velocity: np.ndarray | list[float] | tuple[float, float],
) -> np.ndarray:
    """Return signed normalized normal flow for many event measurements.

    This mirrors ``DVSFullSCGPTracker.signed_normal_flow_for_measurement`` but
    evaluates basis rows, tangents, normals, and velocity projections in array
    form. If the installed PyRecEst backend does not support vectorized basis
    evaluation, the function falls back to the tracker's scalar implementation.
    """

    measurements = np.asarray(event_xy, dtype=float)
    if measurements.ndim == 1 and measurements.size == 0:
        return np.empty(0, dtype=float)
    if measurements.ndim != 2 or measurements.shape[1] != 2:
        raise ValueError("event_xy must have shape (n, 2)")
    if measurements.shape[0] == 0:
        return np.empty(0, dtype=float)

    velocity = np.asarray(event_velocity, dtype=float).reshape(2)
    velocity_norm = float(np.linalg.norm(velocity))
    if velocity_norm <= 1e-12:
        return np.zeros(measurements.shape[0], dtype=float)

    try:
        return _tracker_signed_normal_flows_vectorized_impl(
            tracker,
            measurements,
            velocity,
            velocity_norm,
        )
    except Exception:  # pragma: no cover - backend-specific safety fallback
        return np.asarray(
            [tracker.signed_normal_flow_for_measurement(measurement, velocity) for measurement in measurements],
            dtype=float,
        )


def _tracker_signed_normal_flows_vectorized_impl(
    tracker: DVSFullSCGPTracker,
    measurements: np.ndarray,
    velocity: np.ndarray,
    velocity_norm: float,
) -> np.ndarray:
    position = np.asarray(tracker.kinematic_state[:2], dtype=float)
    orientation = float(tracker.kinematic_state[2])
    shape_state = np.asarray(tracker.shape_state, dtype=float)

    delta = measurements - position[None, :]
    delta_norm = np.linalg.norm(delta, axis=1)
    fallback_direction = np.asarray([np.cos(orientation), np.sin(orientation)], dtype=float)
    unit = np.divide(
        delta,
        delta_norm[:, None],
        out=np.tile(fallback_direction, (measurements.shape[0], 1)),
        where=delta_norm[:, None] > 1e-12,
    )

    world_angles = np.arctan2(unit[:, 1], unit[:, 0])
    body_angles = world_angles - orientation
    basis = np.asarray(tracker._basis_matrix(array(body_angles)), dtype=float)
    basis_derivative = np.asarray(tracker._basis_derivative(array(body_angles)), dtype=float)
    radii = basis @ shape_state
    radius_derivatives = basis_derivative @ shape_state

    tangent = radius_derivatives[:, None] * unit + radii[:, None] * np.column_stack((-unit[:, 1], unit[:, 0]))
    normals = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    normal_norm = np.linalg.norm(normals, axis=1)
    normals = np.divide(
        normals,
        normal_norm[:, None],
        out=np.array(unit, dtype=float, copy=True),
        where=normal_norm[:, None] > 1e-12,
    )
    return (normals @ velocity) / velocity_norm
