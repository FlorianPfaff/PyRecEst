"""Runtime patch for PyTorch reduction-axis normalization."""

from __future__ import annotations

from operator import index as _operator_index

import numpy as np
from pyrecest.backend_support._pytorch_searchsorted_sorter_contract import (
    patch_pytorch_searchsorted_sorter_contract as _patch_pytorch_searchsorted_sorter_contract,
)
from pyrecest.backend_support._pytorch_split_index_contract import (
    patch_pytorch_split_index_contract as _patch_pytorch_split_index_contract,
)

_AXIS_FLAG_TYPES = (bool, np.bool_)
_AXIS_TYPE_MESSAGE = "axis must be an integer or a sequence of integers"


def _is_boolean_axis_value(value, torch_module) -> bool:
    """Return whether ``value`` is a boolean axis scalar/container."""

    if isinstance(value, _AXIS_FLAG_TYPES):
        return True
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.bool_)
    return torch_module.is_tensor(value) and value.dtype == torch_module.bool


def _axis_contains_boolean_value(axis, torch_module) -> bool:
    if axis is None:
        return False
    if _is_boolean_axis_value(axis, torch_module):
        return True
    try:
        _operator_index(axis)
    except TypeError:
        try:
            axes = tuple(axis)
        except TypeError:
            return False
        return any(
            _axis_contains_boolean_value(one_axis, torch_module) for one_axis in axes
        )
    return False


def _normalize_scalar_axis_value(axis, torch_module):
    """Convert integer-like scalar axes to Python integers before dispatch."""

    if axis is None or _is_boolean_axis_value(axis, torch_module):
        return axis
    try:
        return _operator_index(axis)
    except TypeError:
        return axis


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


def _wrap_boolean_axis_reduction(helper, torch_module):
    if getattr(helper, "_pyrecest_reduction_axis_bool_contract", False):
        return helper

    def reduction(*args, **kwargs):
        axis = kwargs.get("axis")
        if "axis" not in kwargs and len(args) >= 2:
            axis = args[1]
        if _axis_contains_boolean_value(axis, torch_module):
            raise TypeError(_AXIS_TYPE_MESSAGE)

        args = list(args)
        kwargs = dict(kwargs)
        if "axis" in kwargs:
            kwargs["axis"] = _normalize_scalar_axis_value(kwargs["axis"], torch_module)
        elif len(args) >= 2:
            args[1] = _normalize_scalar_axis_value(args[1], torch_module)
        return helper(*args, **kwargs)

    reduction.__name__ = getattr(helper, "__name__", "reduction")
    reduction.__doc__ = getattr(helper, "__doc__", None)
    reduction._pyrecest_reduction_axis_bool_contract = True
    return reduction


def _wrap_boolean_axis_dim_reduction(helper, torch_module):
    if getattr(helper, "_pyrecest_reduction_axis_bool_contract", False):
        return helper

    def reduction(*args, **kwargs):
        axis = kwargs.get("axis")
        if "axis" not in kwargs and len(args) >= 2:
            axis = args[1]
        dim = kwargs.get("dim")
        if _axis_contains_boolean_value(
            axis, torch_module
        ) or _axis_contains_boolean_value(dim, torch_module):
            raise TypeError(_AXIS_TYPE_MESSAGE)

        args = list(args)
        kwargs = dict(kwargs)
        if "axis" in kwargs:
            kwargs["axis"] = _normalize_scalar_axis_value(kwargs["axis"], torch_module)
        elif len(args) >= 2:
            args[1] = _normalize_scalar_axis_value(args[1], torch_module)
        if "dim" in kwargs:
            kwargs["dim"] = _normalize_scalar_axis_value(kwargs["dim"], torch_module)
        return helper(*args, **kwargs)

    reduction.__name__ = getattr(helper, "__name__", "reduction")
    reduction.__doc__ = getattr(helper, "__doc__", None)
    reduction._pyrecest_reduction_axis_bool_contract = True
    return reduction


def patch_pytorch_reduction_axis_contract() -> None:
    """Make raw/public PyTorch reduction helpers reject boolean axes."""

    _patch_pytorch_searchsorted_sorter_contract()
    _patch_pytorch_split_index_contract()

    try:
        import pyrecest._backend as backend_loader  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    backend_normalizer = getattr(backend_loader, "_normalize_reduction_axes", None)
    if backend_normalizer is not None and not getattr(
        backend_normalizer,
        "_pyrecest_reduction_axis_bool_contract",
        False,
    ):

        def _normalize_backend_reduction_axes(axis, ndim_value):
            return normalize_reduction_axes(axis, ndim_value, torch)

        _normalize_backend_reduction_axes.__name__ = getattr(
            backend_normalizer,
            "__name__",
            "_normalize_reduction_axes",
        )
        _normalize_backend_reduction_axes.__doc__ = getattr(
            backend_normalizer,
            "__doc__",
            None,
        )
        _normalize_backend_reduction_axes._pyrecest_reduction_axis_bool_contract = True
        backend_loader._normalize_reduction_axes = _normalize_backend_reduction_axes

    original_normalizer = getattr(raw_pytorch, "_normalize_reduction_axes", None)
    if original_normalizer is None:
        return
    if not getattr(
        original_normalizer, "_pyrecest_reduction_axis_bool_contract", False
    ):

        def _normalize_reduction_axes(axis, ndim_value):
            return normalize_reduction_axes(axis, ndim_value, torch)

        _normalize_reduction_axes.__name__ = getattr(
            original_normalizer,
            "__name__",
            "_normalize_reduction_axes",
        )
        _normalize_reduction_axes.__doc__ = getattr(
            original_normalizer, "__doc__", None
        )
        _normalize_reduction_axes._pyrecest_reduction_axis_bool_contract = True
        raw_pytorch._normalize_reduction_axes = _normalize_reduction_axes

    for helper_name in ("any", "all", "count_nonzero", "max", "prod"):
        raw_helper = getattr(raw_pytorch, helper_name, None)
        if raw_helper is not None:
            wrapped_helper = _wrap_boolean_axis_reduction(raw_helper, torch)
            setattr(raw_pytorch, helper_name, wrapped_helper)
            if getattr(backend, "__backend_name__", None) == "pytorch":
                setattr(backend, helper_name, wrapped_helper)

    for helper_name in ("mean", "std", "sum"):
        raw_helper = getattr(raw_pytorch, helper_name, None)
        if raw_helper is not None:
            wrapped_helper = _wrap_boolean_axis_dim_reduction(raw_helper, torch)
            setattr(raw_pytorch, helper_name, wrapped_helper)
            if getattr(backend, "__backend_name__", None) == "pytorch":
                setattr(backend, helper_name, wrapped_helper)
