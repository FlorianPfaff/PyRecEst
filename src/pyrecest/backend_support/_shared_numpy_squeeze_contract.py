"""Shared NumPy/autograd ``squeeze`` compatibility hook."""

from __future__ import annotations

from importlib import import_module
from operator import index as _operator_index


def patch_shared_numpy_squeeze_numpy_contract() -> None:
    """Make shared NumPy ``squeeze`` reject non-singleton requested axes."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    backend_name = getattr(backend, "__backend_name__", None)
    if backend_name not in {"numpy", "autograd"}:
        return

    try:
        import pyrecest._backend._shared_numpy as shared_numpy  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - selected backend import failed
        return

    original_squeeze = shared_numpy.squeeze
    if getattr(original_squeeze, "_pyrecest_nonsingleton_axis_contract", False):
        backend.squeeze = original_squeeze
        raw_backend = import_module(f"pyrecest._backend.{backend_name}")
        raw_backend.squeeze = original_squeeze
        return

    np_module = shared_numpy._np

    def _normalize_axes(axis):
        if isinstance(axis, (int, np_module.integer)):
            return (int(axis),)
        axis_array = np_module.asarray(axis)
        if axis_array.shape == ():
            try:
                return (_operator_index(axis_array),)
            except TypeError as exc:
                raise TypeError(
                    "only integer scalar arrays can be converted to a scalar index"
                ) from exc
        return tuple(axis)

    def _axis_out_of_bounds_error(axis, ndim):
        axis_error = getattr(getattr(np_module, "exceptions", None), "AxisError", None)
        if axis_error is None:
            axis_error = getattr(np_module, "AxisError", None)
        if axis_error is None:
            return ValueError(
                f"axis {axis} is out of bounds for array of dimension {ndim}"
            )
        try:
            return axis_error(axis, ndim=ndim)
        except TypeError:  # pragma: no cover - compatibility with older NumPy APIs
            return axis_error(axis, ndim)

    def squeeze(x, axis=None):
        x_arr = np_module.asarray(x)
        if axis is None:
            return original_squeeze(x_arr, axis=None)

        axes = _normalize_axes(axis)
        if not axes:
            return x_arr

        normalized_axes = []
        for one_axis in axes:
            try:
                one_axis = _operator_index(one_axis)
            except TypeError as exc:
                raise TypeError("axis entries must be integers") from exc
            normalized_axis = one_axis + x_arr.ndim if one_axis < 0 else one_axis
            if normalized_axis < 0 or normalized_axis >= x_arr.ndim:
                raise _axis_out_of_bounds_error(one_axis, x_arr.ndim)
            normalized_axes.append(normalized_axis)
        normalized_axes = tuple(normalized_axes)

        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError("duplicate value in 'axis'")
        if any(x_arr.shape[one_axis] != 1 for one_axis in normalized_axes):
            raise ValueError(
                "cannot select an axis to squeeze out which has size not equal to one"
            )

        squeeze_axis = (
            normalized_axes[0] if len(normalized_axes) == 1 else normalized_axes
        )
        return original_squeeze(x_arr, axis=squeeze_axis)

    squeeze.__name__ = getattr(original_squeeze, "__name__", "squeeze")
    squeeze.__doc__ = getattr(original_squeeze, "__doc__", None)
    squeeze._pyrecest_nonsingleton_axis_contract = True
    squeeze._pyrecest_numpy_contract = True

    shared_numpy.squeeze = squeeze
    backend.squeeze = squeeze
    raw_backend = import_module(f"pyrecest._backend.{backend_name}")
    raw_backend.squeeze = squeeze
