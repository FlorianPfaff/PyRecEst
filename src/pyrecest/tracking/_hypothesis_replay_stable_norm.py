"""Overflow-safe fallback-vector norms for replay hypothesis scoring."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

# pylint: disable=protected-access

_ORIGINAL_ATTR = "_pyrecest_original_finite_record_values"
_MARKER = "_pyrecest_stable_fallback_norm"


def _stable_euclidean_norm(value: Any) -> float:
    """Return a finite vector norm whenever the mathematical result is finite."""

    flattened = np.abs(np.asarray(value, dtype=float).reshape(-1))
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.hypot.reduce(flattened, initial=0.0))


def install_hypothesis_replay_stable_norm_contract() -> None:
    """Prevent finite replay residual vectors from overflowing during scoring."""

    from . import hypothesis_replay as replay_module  # pylint: disable=import-outside-toplevel

    current = replay_module._finite_record_values
    if getattr(current, _MARKER, False):
        return
    if not hasattr(replay_module, _ORIGINAL_ATTR):
        setattr(replay_module, _ORIGINAL_ATTR, current)

    @wraps(current)
    def checked(
        records: Sequence[Any],
        keys: tuple[str, ...],
        *,
        fallback_norm_keys: tuple[str, ...] = (),
        nonnegative: bool = False,
    ) -> np.ndarray:
        values: list[float] = []
        for record in records:
            value = replay_module._record_value(record, keys)
            if value is None and fallback_norm_keys:
                vector = replay_module._record_value(record, fallback_norm_keys)
                if vector is not None and not replay_module._contains_temporal_values(
                    vector
                ):
                    try:
                        value = _stable_euclidean_norm(vector)
                    except (TypeError, ValueError):
                        value = None
            if value is None or replay_module._contains_temporal_values(value):
                continue
            try:
                parsed = float(np.asarray(value, dtype=float))
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(parsed) and (not nonnegative or parsed >= 0.0):
                values.append(parsed)
        return np.asarray(values, dtype=float)

    setattr(checked, _MARKER, True)
    replay_module._finite_record_values = checked


__all__ = ["install_hypothesis_replay_stable_norm_contract"]
