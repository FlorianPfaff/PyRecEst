"""Mask-preserving validation for score-based selection helpers."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from . import selection as _implementation


def _contains_masked_value(value: Any) -> bool:
    """Return whether ``value`` contains genuinely masked NumPy entries."""
    if np.ma.is_masked(value):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return False
        return any(_contains_masked_value(item) for item in value.reshape(-1))
    if isinstance(value, (list, tuple)):
        return any(_contains_masked_value(item) for item in value)
    return False


def install_selection_mask_contract() -> None:
    """Reject masked score values before NumPy exposes their hidden payloads."""
    current = _implementation.sanitized_score_vector
    if getattr(current, "_pyrecest_mask_contract", False):
        return

    @wraps(current)
    def validated_sanitized_score_vector(values, *, nonnegative: bool = True):
        if _contains_masked_value(values):
            raise ValueError("scores must contain real numeric values.")
        return current(values, nonnegative=nonnegative)

    validated_sanitized_score_vector._pyrecest_mask_contract = True
    _implementation.sanitized_score_vector = validated_sanitized_score_vector
