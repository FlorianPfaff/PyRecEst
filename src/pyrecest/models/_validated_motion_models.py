"""Validated wrappers for motion-model catalog helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import motion_models as _motion_models

_nearly_coordinated_turn_model_impl = _motion_models.nearly_coordinated_turn_model
_continuous_to_discrete_lti_impl = _motion_models.continuous_to_discrete_lti
_coordinated_turn_transition_impl = _motion_models.coordinated_turn_transition
_se2_unicycle_transition_impl = _motion_models.se2_unicycle_transition


def _contains_complex_values(value: Any, seen: set[int] | None = None) -> bool:
    """Return whether a possibly nested array-like value contains complex data."""
    if isinstance(value, (complex, np.complexfloating)):
        return True

    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        try:
            if np.dtype(dtype).kind == "c":
                return True
        except TypeError:
            if "complex" in str(dtype).lower():
                return True

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return False
    seen.add(value_id)

    if isinstance(value, (list, tuple)):
        return any(_contains_complex_values(item, seen) for item in value)

    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if np.iscomplexobj(value_array):
        return True
    if value_array.dtype != object:
        return False
    return any(_contains_complex_values(item, seen) for item in value_array.flat)


def _reject_complex_matrix(value: Any, name: str) -> None:
    """Reject complex-valued matrices before NumPy can discard imaginary parts."""
    if _contains_complex_values(value):
        raise ValueError(f"{name} must contain real values")


def continuous_to_discrete_lti(
    continuous_matrix: Any,
    noise_input_matrix: Any | None = None,
    continuous_noise_covariance: Any | None = None,
    dt: float = 1.0,
) -> Any:
    """Discretize a real LTI model without silently truncating complex inputs."""
    _reject_complex_matrix(continuous_matrix, "continuous_matrix")
    if noise_input_matrix is not None:
        _reject_complex_matrix(noise_input_matrix, "noise_input_matrix")
    if continuous_noise_covariance is not None:
        _reject_complex_matrix(
            continuous_noise_covariance,
            "continuous_noise_covariance",
        )
    return _continuous_to_discrete_lti_impl(
        continuous_matrix,
        noise_input_matrix,
        continuous_noise_covariance,
        dt=dt,
    )


def coordinated_turn_transition(
    state: Any, dt: float = 1.0, turn_threshold: float = 1e-8
) -> Any:
    """Propagate a coordinated-turn state with a valid branch threshold."""
    turn_threshold = (
        _motion_models._as_positive_float(  # pylint: disable=protected-access
            turn_threshold,
            "turn_threshold",
        )
    )
    return _coordinated_turn_transition_impl(
        state,
        dt=dt,
        turn_threshold=turn_threshold,
    )


def se2_unicycle_transition(
    state: Any, dt: float = 1.0, turn_threshold: float = 1e-8
) -> Any:
    """Propagate an SE(2) unicycle state with a valid branch threshold."""
    turn_threshold = (
        _motion_models._as_positive_float(  # pylint: disable=protected-access
            turn_threshold,
            "turn_threshold",
        )
    )
    return _se2_unicycle_transition_impl(
        state,
        dt=dt,
        turn_threshold=turn_threshold,
    )


def nearly_coordinated_turn_model(
    dt: float = 1.0,
    position_spectral_density: float = 1.0,
    turn_rate_variance: float = 1e-4,
) -> Any:
    """Return a coordinated-turn model with validated nearly-constant-turn covariance."""
    dt = _motion_models._as_nonnegative_float(  # pylint: disable=protected-access
        dt,
        "dt",
    )
    turn_rate_variance = (
        _motion_models._as_nonnegative_float(  # pylint: disable=protected-access
            turn_rate_variance,
            "turn_rate_variance",
        )
    )
    return _nearly_coordinated_turn_model_impl(
        dt=dt,
        position_spectral_density=position_spectral_density,
        turn_rate_variance=turn_rate_variance,
    )


_motion_models.continuous_to_discrete_lti = continuous_to_discrete_lti
_motion_models.coordinated_turn_transition = coordinated_turn_transition
_motion_models.nearly_coordinated_turn_model = nearly_coordinated_turn_model
_motion_models.se2_unicycle_transition = se2_unicycle_transition


__all__ = [
    "continuous_to_discrete_lti",
    "coordinated_turn_transition",
    "nearly_coordinated_turn_model",
    "se2_unicycle_transition",
]
