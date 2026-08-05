"""Overflow-safe bounding-box mapping for Euclidean rejection sampling."""

from __future__ import annotations

from functools import wraps

import numpy as np

from . import euclidean_sampler as _euclidean_sampler

_ORIGINAL_ATTR = "_pyrecest_original_map_to_bounding_box"


def _map_extreme_bounding_box(unit_samples, bounding_box):
    """Map unit samples by convex endpoint interpolation without a raw span."""

    unit_samples = np.asarray(unit_samples)
    bounding_box = np.asarray(bounding_box)
    lower = bounding_box[:, 0]
    upper = bounding_box[:, 1]
    return (1.0 - unit_samples) * lower + unit_samples * upper


def install_fibonacci_rejection_extreme_bounds_contract() -> None:
    """Preserve finite candidates when a finite bounding-box span overflows."""

    sampler_type = _euclidean_sampler.FibonacciRejectionSampler
    current = sampler_type._map_to_bounding_box
    if getattr(current, "_pyrecest_extreme_bounds_safe", False):
        return

    if not hasattr(sampler_type, _ORIGINAL_ATTR):
        setattr(sampler_type, _ORIGINAL_ATTR, current)
    original = getattr(sampler_type, _ORIGINAL_ATTR)

    @wraps(original)
    def checked(unit_samples, bounding_box):
        lower = bounding_box[:, 0]
        upper = bounding_box[:, 1]
        with np.errstate(over="ignore", invalid="ignore"):
            width = upper - lower
        if np.all(np.isfinite(width)):
            return original(unit_samples, bounding_box)
        return _map_extreme_bounding_box(unit_samples, bounding_box)

    checked._pyrecest_extreme_bounds_safe = True
    sampler_type._map_to_bounding_box = staticmethod(checked)
