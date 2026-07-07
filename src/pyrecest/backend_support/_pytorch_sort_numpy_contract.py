"""PyTorch sort helpers for backend compatibility."""

from __future__ import annotations

from operator import index as _operator_index

import numpy as _np

_SORT_KIND_MESSAGE = (
    "sort kind must be one of 'quicksort', 'heapsort', 'stable', or 'mergesort'"
)
_SORT_CONFLICT_MESSAGE = "sort() got both 'kind' and 'stable' arguments"


def normalize_sort_axis(axis):
    """Return a sort axis while preserving NumPy's flatten-all sentinel."""
    if axis is None:
        return None
    if isinstance(axis, (bool, _np.bool_)):
        raise TypeError("an integer is required for the axis")
    return _operator_index(axis)


def resolve_sort_stability(kind, stable):
    """Return the torch ``stable`` flag implied by NumPy-style options."""
    # NumPy treats ``kind`` and ``stable`` as mutually exclusive sort-mode
    # selectors, even when they would imply the same stable/unstable choice.
    if kind is not None and stable is not None:
        raise ValueError(_SORT_CONFLICT_MESSAGE)
    if kind is None:
        return stable
    if kind in {"stable", "mergesort"}:
        return True
    if kind in {"quicksort", "heapsort"}:
        return False
    raise ValueError(_SORT_KIND_MESSAGE)


def sort_axis_none(
    backend_module,
    torch_module,
    values,
    axis=-1,
    kind=None,
    order=None,
    *,
    stable=None,
    descending=False,
):
    """Sort values with NumPy-style ``axis=None`` and keyword support."""
    if order is not None:
        raise ValueError("order is not supported by this backend")
    values = backend_module.array(values)
    axis = normalize_sort_axis(axis)
    stable = resolve_sort_stability(kind, stable)
    if axis is None:
        values = torch_module.flatten(values)
        axis = 0
    return torch_module.sort(
        values,
        dim=axis,
        descending=bool(descending),
        stable=bool(stable) if stable is not None else False,
    ).values
