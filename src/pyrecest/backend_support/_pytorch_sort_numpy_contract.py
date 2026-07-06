"""PyTorch sort helpers for backend compatibility."""

from __future__ import annotations

from operator import index as _operator_index


def normalize_sort_axis(axis):
    """Return a sort axis while preserving NumPy's flatten-all sentinel."""
    if axis is None:
        return None
    return _operator_index(axis)


def flatten_axis_for_sort(axis):
    """Return zero for flattened sorting and a normalized axis otherwise."""
    normalized_axis = normalize_sort_axis(axis)
    if normalized_axis is None:
        return 0
    return normalized_axis
