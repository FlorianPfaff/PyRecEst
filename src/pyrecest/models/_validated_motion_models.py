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


def _singer_coupling_factors(normalized_interval: float) -> tuple[float, float]:
    """Return stable acceleration-to-position and acceleration-to-velocity factors."""

    x = normalized_interval
    if abs(x) < 1.0e-4:
        x_squared = x * x
        velocity_factor = (
            1.0
            - 0.5 * x
            + x_squared / 6.0
            - x_squared * x / 24.0
            + x_squared * x_squared / 120.0
        )
        position_factor = (
            0.5
            - x / 6.0
            + x_squared / 24.0
            - x_squared * x / 120.0
            + x_squared * x_squared / 720.0
        )
        return position_factor, velocity_factor

    expm1_negative_x = np.expm1(-x)
    velocity_factor = -expm1_negative_x / x
    position_factor = (x + expm1_negative_x) / (x * x)
    return position_factor, velocity_factor


def singer_transition_matrix(
    dt: float, spatial_dim: int = 2, tau: float = 20.0
) -> Any:
    """Return a numerically stable Singer acceleration transition matrix."""

    dt = _motion_models._as_scalar_float(  # pylint: disable=protected-access
        dt,
        "dt",
    )
    spatial_dim = _motion_models._as_positive_integer(  # pylint: disable=protected-access
        spatial_dim,
        "spatial_dim",
    )
    tau = _motion_models._as_positive_float(  # pylint: disable=protected-access
        tau,
        "tau",
    )

    normalized_interval = dt / tau
    position_factor, velocity_factor = _singer_coupling_factors(
        normalized_interval
    )
    block = np.array(
        [
            [1.0, dt, dt * dt * position_factor],
            [0.0, 1.0, dt * velocity_factor],
            [0.0, 0.0, np.exp(-normalized_interval)],
        ],
        dtype=float,
    )
    matrix = np.zeros((3 * spatial_dim, 3 * spatial_dim), dtype=float)
    for row_derivative in range(3):
        for col_derivative in range(3):
            for axis in range(spatial_dim):
                matrix[
                    _motion_models._state_index(  # pylint: disable=protected-access
                        row_derivative,
                        axis,
                        spatial_dim,
                    ),
                    _motion_models._state_index(  # pylint: disable=protected-access
                        col_derivative,
                        axis,
                        spatial_dim,
                    ),
                ] = block[row_derivative, col_derivative]
    return _motion_models.asarray(matrix)


_motion_models.continuous_to_discrete_lti = continuous_to_discrete_lti
_motion_models.coordinated_turn_transition = coordinated_turn_transition
_motion_models.nearly_coordinated_turn_model = nearly_coordinated_turn_model
_motion_models.se2_unicycle_transition = se2_unicycle_transition
_motion_models.singer_transition_matrix = singer_transition_matrix


__all__ = [
    "continuous_to_discrete_lti",
    "coordinated_turn_transition",
    "nearly_coordinated_turn_model",
    "se2_unicycle_transition",
    "singer_transition_matrix",
]
