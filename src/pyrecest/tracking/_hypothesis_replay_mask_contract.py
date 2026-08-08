"""Mask-preserving validation for hypothesis replay inputs."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

from . import hypothesis_replay as _implementation

# The contract deliberately wraps private implementation helpers.
# pylint: disable=protected-access

_ORIGINAL_FINITE_RECORD_VALUES = _implementation._finite_record_values
_ORIGINAL_FINITE_FLOAT = _implementation._finite_float
_ORIGINAL_NONNEGATIVE_INT = _implementation._nonnegative_int
_MARKER = "_pyrecest_mask_preserving_hypothesis_replay_contract"


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


@wraps(_ORIGINAL_FINITE_RECORD_VALUES)
def _finite_record_values(
    records: Sequence[Any],
    keys: tuple[str, ...],
    *,
    fallback_norm_keys: tuple[str, ...] = (),
    nonnegative: bool = False,
) -> np.ndarray:
    """Collect valid replay statistics without exposing masked payloads."""

    values: list[float] = []
    for record in records:
        value = _implementation._record_value(record, keys)
        if value is None and fallback_norm_keys:
            vector = _implementation._record_value(record, fallback_norm_keys)
            if (
                vector is not None
                and not _contains_masked_value(vector)
                and not _implementation._contains_temporal_values(vector)
            ):
                try:
                    value = float(
                        np.linalg.norm(np.asarray(vector, dtype=float).reshape(-1))
                    )
                except (TypeError, ValueError):
                    value = None
        if (
            value is None
            or _contains_masked_value(value)
            or _implementation._contains_temporal_values(value)
        ):
            continue
        try:
            parsed = float(np.asarray(value, dtype=float))
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(parsed) and (not nonnegative or parsed >= 0.0):
            values.append(parsed)
    return np.asarray(values, dtype=float)


@wraps(_ORIGINAL_FINITE_FLOAT)
def _finite_float(value: Any, name: str) -> float:
    """Validate scalar replay controls without discarding NumPy masks."""

    if _contains_masked_value(value):
        raise ValueError(f"{name} must be finite")
    return _ORIGINAL_FINITE_FLOAT(value, name)


@wraps(_ORIGINAL_NONNEGATIVE_INT)
def _nonnegative_int(value: Any, name: str) -> int:
    """Validate integer replay controls without discarding NumPy masks."""

    if _contains_masked_value(value):
        raise ValueError(f"{name} must be a nonnegative integer")
    return _ORIGINAL_NONNEGATIVE_INT(value, name)


def install_hypothesis_replay_mask_contract() -> None:
    """Install mask-preserving replay validation exactly once."""

    setattr(_finite_record_values, _MARKER, True)
    setattr(_finite_float, _MARKER, True)
    setattr(_nonnegative_int, _MARKER, True)

    if not getattr(_implementation._finite_record_values, _MARKER, False):
        _implementation._finite_record_values = _finite_record_values
    if not getattr(_implementation._finite_float, _MARKER, False):
        _implementation._finite_float = _finite_float
    if not getattr(_implementation._nonnegative_int, _MARKER, False):
        _implementation._nonnegative_int = _nonnegative_int


__all__ = ["install_hypothesis_replay_mask_contract"]
