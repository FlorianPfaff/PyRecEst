"""Utilities for exposing virtual backend submodules."""

from __future__ import annotations

import sys
from functools import wraps
from operator import index as _operator_index
from types import ModuleType

from pyrecest._backend import BACKEND_ATTRIBUTES


def _copy_result_to_out(result, out):
    """Copy ``result`` into a backend ``out`` object and return that object."""
    copy_ = getattr(out, "copy_", None)
    if copy_ is not None:
        copy_(result)
        return out
    try:
        out[...] = result
    except TypeError:
        at = getattr(out, "at", None)
        if at is None:
            raise
        return at[...].set(result)
    return out


def _cumulative_with_out(cumulative):
    """Return a cumulative helper accepting NumPy's ``out`` keyword."""

    @wraps(cumulative)
    def wrapped_cumulative(x, axis=None, dtype=None, out=None):
        result = cumulative(x, axis=axis, dtype=dtype)
        if out is not None:
            return _copy_result_to_out(result, out)
        return result

    wrapped_cumulative._pyrecest_out_contract = True
    return wrapped_cumulative


def _adapt_cumulative_out_contract(backend: ModuleType) -> None:
    """Adapt PyTorch cumulative helpers to the public NumPy-style contract."""
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return
    for attribute_name in ("cumsum", "cumprod"):
        cumulative = getattr(backend, attribute_name, None)
        if cumulative is None or getattr(cumulative, "_pyrecest_out_contract", False):
            continue
        setattr(backend, attribute_name, _cumulative_with_out(cumulative))


def _adapt_pytorch_repeat_contract(backend: ModuleType) -> None:
    """Adapt PyTorch repeat to NumPy's ``axis`` keyword contract."""
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    current_repeat = getattr(backend, "repeat", None)
    if current_repeat is not None and getattr(
        current_repeat, "_pyrecest_axis_contract", False
    ):
        return

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - backend import fails first in practice
        return

    integer_dtypes = {
        _torch.uint8,
        _torch.int8,
        _torch.int16,
        _torch.int32,
        _torch.int64,
    }

    def _repeat_count(repeats, *, device):
        if _torch.is_tensor(repeats):
            repeats_tensor = repeats.to(device=device)
        else:
            repeats_tensor = _torch.as_tensor(repeats, device=device)

        if repeats_tensor.ndim == 0:
            try:
                return _operator_index(repeats_tensor.item())
            except TypeError as exc:
                raise TypeError("repeats must be integers") from exc

        if repeats_tensor.dtype not in integer_dtypes:
            raise TypeError("repeats must be integers")
        return repeats_tensor.to(dtype=_torch.long)

    def _repeat_axis(axis, ndim):
        axis = _operator_index(axis)
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise IndexError(f"axis {axis} is out of bounds for array of dimension {ndim}")
        return axis

    def repeat(a, repeats, axis=None, *, dim=None):
        if dim is not None:
            if axis is not None and axis != dim:
                raise TypeError("repeat() got both 'axis' and 'dim'")
            axis = dim

        values = backend.array(a)
        if axis is None:
            values = values.flatten()
            axis = 0
        else:
            axis = _repeat_axis(axis, values.ndim)

        repeat_count = _repeat_count(repeats, device=values.device)
        return _torch.repeat_interleave(values, repeat_count, dim=axis)

    repeat.__name__ = "repeat"
    repeat.__doc__ = getattr(_torch.repeat_interleave, "__doc__", None)
    repeat._pyrecest_axis_contract = True
    backend.repeat = repeat
    pytorch_backend.repeat = repeat


def register_backend_submodules(backend: ModuleType | None = None) -> None:
    """Register virtual backend submodules for standard import statements."""
    if backend is None:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    _adapt_cumulative_out_contract(backend)
    _adapt_pytorch_repeat_contract(backend)

    backend.__path__ = getattr(backend, "__path__", [])
    backend_spec = getattr(backend, "__spec__", None)
    if backend_spec is not None:
        backend_spec.submodule_search_locations = (
            getattr(backend_spec, "submodule_search_locations", None) or []
        )

    for submodule_name in BACKEND_ATTRIBUTES:
        if not submodule_name:
            continue
        sys.modules[f"{backend.__name__}.{submodule_name}"] = getattr(
            backend, submodule_name
        )
