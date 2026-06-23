"""Validated wrappers for motion-model catalog helpers."""

from __future__ import annotations

from typing import Any

from . import motion_models as _motion_models

_nearly_coordinated_turn_model_impl = _motion_models.nearly_coordinated_turn_model


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


_motion_models.nearly_coordinated_turn_model = nearly_coordinated_turn_model


__all__ = ["nearly_coordinated_turn_model"]
