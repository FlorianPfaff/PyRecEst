"""Active-contour helpers for DVS extended-object observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RectangleContour:
    """Sampled rectangle boundary with outward normals and edge labels."""

    points: np.ndarray
    normals: np.ndarray
    edge_labels: list[str]


def unit_vector_from_angle(angle: float) -> np.ndarray:
    """Return the two-dimensional unit vector for an angle in radians."""
    return np.array([np.cos(angle), np.sin(angle)], dtype=float)


def signed_normal_flow(normal: np.ndarray, velocity: np.ndarray) -> float:
    """Return signed normalized normal flow for one contour normal.

    The sign is the part that event polarity can constrain. The unsigned
    activity model remains ``abs(signed_normal_flow(...))``.
    """
    normal = np.asarray(normal, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    velocity_norm = np.linalg.norm(velocity)
    normal_norm = np.linalg.norm(normal)
    if velocity_norm <= 0.0 or normal_norm <= 0.0:
        return 0.0
    return float((normal / normal_norm) @ velocity / velocity_norm)


def normal_flow_activity(normal: np.ndarray, velocity: np.ndarray) -> float:
    """Return normalized normal-flow activity for one contour normal."""
    return abs(signed_normal_flow(normal, velocity))


def signed_normal_flow_profile(normals: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Evaluate signed normalized normal flow for multiple contour normals."""
    return np.array(
        [signed_normal_flow(normal, velocity) for normal in normals],
        dtype=float,
    )


def activity_profile(
    normals: np.ndarray,
    velocity: np.ndarray,
    activity_floor: float | None = None,
) -> np.ndarray:
    """Evaluate normalized normal-flow activity for multiple contour normals."""
    activities = np.abs(signed_normal_flow_profile(normals, velocity))
    if activity_floor is not None:
        activities = np.maximum(activities, float(activity_floor))
    return activities


def rectangle_contour_samples(
    width: float = 2.0,
    height: float = 1.0,
    samples_per_edge: int = 40,
) -> RectangleContour:
    """Sample a rectangle contour with outward normals.

    This is a simple image-plane proxy for the cube translation example: for
    horizontal motion, the left and right sides are active while top and bottom
    sides are inactive.
    """
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive")
    if samples_per_edge <= 0:
        raise ValueError("samples_per_edge must be positive")

    half_width = 0.5 * float(width)
    half_height = 0.5 * float(height)
    xs = np.linspace(-half_width, half_width, samples_per_edge, endpoint=False)
    ys = np.linspace(-half_height, half_height, samples_per_edge, endpoint=False)

    top = np.column_stack([xs, np.full(samples_per_edge, half_height)])
    right = np.column_stack([np.full(samples_per_edge, half_width), ys])
    bottom = np.column_stack([xs[::-1], np.full(samples_per_edge, -half_height)])
    left = np.column_stack([np.full(samples_per_edge, -half_width), ys[::-1]])

    points = np.vstack([top, right, bottom, left])
    normals = np.vstack(
        [
            np.tile([0.0, 1.0], (samples_per_edge, 1)),
            np.tile([1.0, 0.0], (samples_per_edge, 1)),
            np.tile([0.0, -1.0], (samples_per_edge, 1)),
            np.tile([-1.0, 0.0], (samples_per_edge, 1)),
        ]
    )
    edge_labels = (
        ["top"] * samples_per_edge
        + ["right"] * samples_per_edge
        + ["bottom"] * samples_per_edge
        + ["left"] * samples_per_edge
    )
    return RectangleContour(points=points, normals=normals, edge_labels=edge_labels)
