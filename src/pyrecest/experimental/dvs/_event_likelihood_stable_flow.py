"""Overflow-safe normal-flow activity contract for DVS event likelihoods."""

from __future__ import annotations

from functools import wraps

import numpy as np

from . import event_likelihood as _event_likelihood

# pylint: disable=protected-access

_MARKER = "_pyrecest_overflow_safe_normal_flow_activities"


def _stable_euclidean_norm(values: np.ndarray, *, axis=None):
    """Return Euclidean norms without overflowing intermediate squares."""

    return np.hypot.reduce(np.abs(values), axis=axis, initial=0.0)


def install_event_likelihood_stable_flow_contract() -> None:
    """Preserve finite DVS normal-flow directions at extreme magnitudes."""

    current = _event_likelihood.normal_flow_activities
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def stable_normal_flow_activities(
        normals: np.ndarray,
        velocity: np.ndarray,
        activity_floor: float = 0.0,
    ) -> np.ndarray:
        normals = np.asarray(normals, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        activity_floor = _event_likelihood._validate_nonnegative_finite(
            activity_floor,
            "activity_floor",
        )
        if normals.ndim != 2 or normals.shape[1] != 2:
            raise ValueError("normals must have shape (n, 2)")
        if velocity.shape != (2,):
            raise ValueError("velocity must have shape (2,)")
        _event_likelihood._validate_finite_array(normals, "normals")
        _event_likelihood._validate_finite_array(velocity, "velocity")

        velocity_norm = float(_stable_euclidean_norm(velocity))
        if velocity_norm <= 1e-12:
            activities = np.zeros(normals.shape[0], dtype=float)
        else:
            normal_norms = _stable_euclidean_norm(normals, axis=1)
            unit_normals = np.divide(
                normals,
                normal_norms[:, None],
                out=np.zeros_like(normals),
                where=normal_norms[:, None] > 1e-12,
            )
            unit_velocity = velocity / velocity_norm
            activities = np.abs(unit_normals @ unit_velocity)
        if activity_floor > 0.0:
            activities = np.maximum(activities, activity_floor)
        return activities

    setattr(stable_normal_flow_activities, _MARKER, True)
    _event_likelihood.normal_flow_activities = stable_normal_flow_activities


__all__ = ["install_event_likelihood_stable_flow_contract"]
