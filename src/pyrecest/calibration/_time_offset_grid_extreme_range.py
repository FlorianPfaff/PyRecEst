"""Overflow-safe contracts for time-offset calibration."""

from __future__ import annotations

from functools import wraps

import numpy as np

from . import time_offset as _time_offset
from ._time_offset_stable_statistics import (
    install_time_offset_stable_statistics_contract,
)

_ORIGINAL_ATTR = "_pyrecest_original_make_offset_grid"


def _extreme_range_grid(
    min_s: float, max_s: float, step_s: float
) -> np.ndarray:
    original = getattr(_time_offset, _ORIGINAL_ATTR)
    min_s = _time_offset._as_finite_float(min_s, "min_s")
    max_s = _time_offset._as_finite_float(max_s, "max_s")
    step_s = _time_offset._as_finite_float(step_s, "step_s")
    if step_s <= 0.0:
        raise ValueError("step_s must be positive")
    if max_s < min_s:
        raise ValueError("max_s must be greater than or equal to min_s")

    with np.errstate(over="ignore", invalid="ignore"):
        span_s = max_s - min_s
    if np.isfinite(span_s):
        return original(min_s, max_s, step_s)

    # Dividing each opposite-sign endpoint before combining their magnitudes
    # avoids overflowing the raw span. It also keeps the subsequent grid
    # construction in bounded step units.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        min_steps = min_s / step_s
        span_steps = -min_steps + max_s / step_s
    if not np.isfinite(span_steps) or span_steps > np.iinfo(np.intp).max - 1:
        raise ValueError("offset grid would contain too many points")

    count = int(np.floor(span_steps)) + 1
    offsets = (min_steps + np.arange(count, dtype=float)) * step_s
    offsets[0] = min_s
    if offsets.size == 0 or offsets[-1] < max_s - 1.0e-12:
        offsets = np.append(offsets, max_s)
    return offsets


def install_time_offset_grid_extreme_range_contract() -> None:
    """Install overflow-safe grid and statistic handling."""

    install_time_offset_stable_statistics_contract()
    if not hasattr(_time_offset, _ORIGINAL_ATTR):
        setattr(_time_offset, _ORIGINAL_ATTR, _time_offset.make_offset_grid)
    if getattr(_time_offset.make_offset_grid, "_pyrecest_extreme_range_safe", False):
        return

    original = getattr(_time_offset, _ORIGINAL_ATTR)

    @wraps(original)
    def checked(min_s: float, max_s: float, step_s: float) -> np.ndarray:
        return _extreme_range_grid(min_s, max_s, step_s)

    setattr(checked, "_pyrecest_extreme_range_safe", True)
    _time_offset.make_offset_grid = checked
