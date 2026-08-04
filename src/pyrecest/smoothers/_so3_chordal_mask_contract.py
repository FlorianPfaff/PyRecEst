"""Mask-preserving validation for SO(3) chordal smoothing inputs."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from . import so3_chordal_mean_smoother as _implementation

_MARKER = "_pyrecest_so3_chordal_mask_contract"


def _contains_masked_value(
    value: Any,
    active_ids: set[int] | None = None,
) -> bool:
    """Return whether an array-like value contains genuinely masked data."""

    if np.ma.is_masked(value):
        return True
    if not isinstance(value, (np.ndarray, list, tuple)):
        return False

    if active_ids is None:
        active_ids = set()
    value_id = id(value)
    if value_id in active_ids:
        return False
    active_ids.add(value_id)
    try:
        if isinstance(value, np.ndarray):
            if value.dtype != object:
                return False
            values = value.flat
        else:
            values = value
        return any(_contains_masked_value(item, active_ids) for item in values)
    finally:
        active_ids.remove(value_id)


def install_so3_chordal_mask_contract() -> None:
    """Reject masked SO(3) smoother inputs before backend coercion."""

    smoother_type = _implementation.SO3ChordalMeanSmoother

    current_window_validator = smoother_type._validate_window_size
    if not getattr(current_window_validator, _MARKER, False):

        @wraps(current_window_validator)
        def validated_window_size(window_size: int) -> int:
            if _contains_masked_value(window_size):
                raise ValueError("window_size must be a positive integer.")
            return current_window_validator(window_size)

        setattr(validated_window_size, _MARKER, True)
        smoother_type._validate_window_size = staticmethod(validated_window_size)

    current_weight_validator = smoother_type._normalize_weight_vector
    if not getattr(current_weight_validator, _MARKER, False):

        @wraps(current_weight_validator)
        def validated_weight_vector(
            weights,
            length: int,
            name: str,
            normalize: bool = True,
        ):
            if _contains_masked_value(weights):
                raise ValueError(f"{name} must not contain masked values.")
            return current_weight_validator(weights, length, name, normalize)

        setattr(validated_weight_vector, _MARKER, True)
        smoother_type._normalize_weight_vector = staticmethod(validated_weight_vector)

    current_rotation_parser = smoother_type._as_rotation_list
    if not getattr(current_rotation_parser, _MARKER, False):

        @wraps(current_rotation_parser)
        def validated_rotation_list(rotations) -> list:
            if _contains_masked_value(rotations):
                raise ValueError("rotations must not contain masked values.")
            return current_rotation_parser(rotations)

        setattr(validated_rotation_list, _MARKER, True)
        smoother_type._as_rotation_list = staticmethod(validated_rotation_list)

    current_projection = smoother_type.project_to_so3
    if not getattr(current_projection, _MARKER, False):

        @wraps(current_projection)
        def validated_projection(matrix):
            if _contains_masked_value(matrix):
                raise ValueError("matrix must not contain masked values.")
            return current_projection(matrix)

        setattr(validated_projection, _MARKER, True)
        smoother_type.project_to_so3 = staticmethod(validated_projection)

    current_distance = smoother_type.chordal_distance
    if not getattr(current_distance, _MARKER, False):

        @wraps(current_distance)
        def validated_distance(rotation_a, rotation_b):
            if _contains_masked_value(rotation_a) or _contains_masked_value(rotation_b):
                raise ValueError("rotations must not contain masked values.")
            return current_distance(rotation_a, rotation_b)

        setattr(validated_distance, _MARKER, True)
        smoother_type.chordal_distance = staticmethod(validated_distance)


__all__ = ["install_so3_chordal_mask_contract"]
