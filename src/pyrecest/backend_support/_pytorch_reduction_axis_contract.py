"""Runtime patch for PyTorch reduction-axis normalization."""

from __future__ import annotations

from operator import index as _operator_index

import numpy as np

_AXIS_FLAG_TYPES = (bool, np.bool_)
_AXIS_TYPE_MESSAGE = "axis must be an integer or a sequence of integers"


def _is_boolean_axis_value(value, torch_module) -> bool:
    """Return whether ``value`` is a boolean axis scalar/container."""

    if isinstance(value, _AXIS_FLAG_TYPES):
        return True
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.bool_)
    return torch_module.is_tensor(value) and value.dtype == torch_module.bool


def normalize_reduction_axes(axis, ndim_value, torch_module):
    """Normalize PyTorch reduction axes while rejecting boolean axes."""

    if _is_boolean_axis_value(axis, torch_module):
        raise TypeError(_AXIS_TYPE_MESSAGE)
    try:
        axis_index = _operator_index(axis)
    except TypeError as index_error:
        if getattr(axis, "shape", None) == ():
            raise TypeError(_AXIS_TYPE_MESSAGE) from index_error
        try:
            axes = tuple(axis)
        except TypeError as iterable_error:
            raise TypeError(_AXIS_TYPE_MESSAGE) from iterable_error
        if any(_is_boolean_axis_value(one_axis, torch_module) for one_axis in axes):
            raise TypeError(_AXIS_TYPE_MESSAGE)
        try:
            axes = tuple(_operator_index(one_axis) for one_axis in axes)
        except TypeError as item_error:
            raise TypeError(_AXIS_TYPE_MESSAGE) from item_error
    else:
        axes = (axis_index,)

    normalized_axes = tuple(
        one_axis + ndim_value if one_axis < 0 else one_axis for one_axis in axes
    )
    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError("duplicate value in 'axis'")

    for original_axis, normalized_axis in zip(axes, normalized_axes):
        if normalized_axis < 0 or normalized_axis >= ndim_value:
            raise IndexError(
                f"axis {original_axis} is out of bounds for array of dimension {ndim_value}"
            )

    return normalized_axes


def patch_pytorch_reduction_axis_contract() -> None:
    """Make raw/public PyTorch reduction helpers reject boolean axes."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_normalizer = getattr(raw_pytorch, "_normalize_reduction_axes", None)
    if original_normalizer is None:
        return
    if getattr(original_normalizer, "_pyrecest_reduction_axis_bool_contract", False):
        return

    def _normalize_reduction_axes(axis, ndim_value):
        return normalize_reduction_axes(axis, ndim_value, torch)

    _normalize_reduction_axes.__name__ = getattr(
        original_normalizer,
        "__name__",
        "_normalize_reduction_axes",
    )
    _normalize_reduction_axes.__doc__ = getattr(original_normalizer, "__doc__", None)
    _normalize_reduction_axes._pyrecest_reduction_axis_bool_contract = True
    raw_pytorch._normalize_reduction_axes = _normalize_reduction_axes
