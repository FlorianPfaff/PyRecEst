"""Reject masked support-point inputs before NumPy coercion."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import support_points as _support_points

_ORIGINAL_REAL_ARRAY_ATTR = "_pyrecest_original_support_points_as_real_array"
_ORIGINAL_NONNEGATIVE_SCALAR_ATTR = (
    "_pyrecest_original_support_points_as_finite_nonnegative_scalar"
)
_ORIGINAL_BOOL_SCALAR_ATTR = "_pyrecest_original_support_points_as_bool_scalar"


def _contains_masked_values(
    value: Any,
    active_ids: set[int] | None = None,
) -> bool:
    """Return whether a scalar or nested array-like contains a real mask."""

    if np.ma.is_masked(value):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return False
        items = value.reshape(-1)
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return False

    if active_ids is None:
        active_ids = set()
    value_id = id(value)
    if value_id in active_ids:
        return False
    active_ids.add(value_id)
    try:
        return any(_contains_masked_values(item, active_ids) for item in items)
    finally:
        active_ids.remove(value_id)


def install_support_points_masked_input_contract() -> None:
    """Install mask-aware wrappers for shared support-point validators."""

    if not hasattr(_support_points, _ORIGINAL_REAL_ARRAY_ATTR):
        setattr(
            _support_points,
            _ORIGINAL_REAL_ARRAY_ATTR,
            _support_points._as_real_array,
        )
    if not hasattr(_support_points, _ORIGINAL_NONNEGATIVE_SCALAR_ATTR):
        setattr(
            _support_points,
            _ORIGINAL_NONNEGATIVE_SCALAR_ATTR,
            _support_points._as_finite_nonnegative_scalar,
        )
    if not hasattr(_support_points, _ORIGINAL_BOOL_SCALAR_ATTR):
        setattr(
            _support_points,
            _ORIGINAL_BOOL_SCALAR_ATTR,
            _support_points._as_bool_scalar,
        )

    base_real_array = getattr(_support_points, _ORIGINAL_REAL_ARRAY_ATTR)
    base_nonnegative_scalar = getattr(
        _support_points,
        _ORIGINAL_NONNEGATIVE_SCALAR_ATTR,
    )
    base_bool_scalar = getattr(_support_points, _ORIGINAL_BOOL_SCALAR_ATTR)

    def _as_real_array(name: str, value: Any) -> np.ndarray:
        if _contains_masked_values(value):
            raise ValueError(f"{name} must contain real numeric values.")
        return base_real_array(name, value)

    def _as_finite_nonnegative_scalar(name: str, value: Any) -> float:
        if _contains_masked_values(value):
            raise ValueError(f"{name} must be a finite non-negative scalar.")
        return base_nonnegative_scalar(name, value)

    def _as_bool_scalar(name: str, value: Any) -> bool:
        if _contains_masked_values(value):
            raise ValueError(f"{name} must be a boolean.")
        return base_bool_scalar(name, value)

    _support_points._as_real_array = _as_real_array
    _support_points._as_finite_nonnegative_scalar = _as_finite_nonnegative_scalar
    _support_points._as_bool_scalar = _as_bool_scalar


__all__ = ["install_support_points_masked_input_contract"]
