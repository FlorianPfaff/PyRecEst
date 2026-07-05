"""PyTorch NumPy compatibility hooks for raw/public array helpers."""

from __future__ import annotations

from operator import index as _operator_index


def _preferred_pytorch_device(torch_module, *values):
    """Return an existing non-CPU tensor device, falling back to any tensor."""
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def _as_trapezoid_tensor(value, torch_module, *, device=None, dtype=None):
    """Coerce one trapezoid argument without moving existing tensors unnecessarily."""
    if torch_module.is_tensor(value):
        target_device = device if device is not None else value.device
        target_dtype = dtype if dtype is not None else value.dtype
        if value.device != target_device or value.dtype != target_dtype:
            return value.to(device=target_device, dtype=target_dtype)
        return value
    return torch_module.as_tensor(value, device=device, dtype=dtype)


def _promote_trapezoid_tensor(value, raw_pytorch):
    """Promote integer and boolean inputs before PyTorch integration."""
    if value.dtype.is_floating_point or value.dtype.is_complex:
        return value
    return value.to(dtype=raw_pytorch.get_default_dtype())


def _split_cut_index(index) -> int:
    """Return one NumPy-style split cut index without lossy coercion."""
    try:
        return _operator_index(index)
    except TypeError as exc:
        raise TypeError(
            "slice indices must be integers or None or have an __index__ method"
        ) from exc


def _slice_along_axis(x, start, stop, axis):
    index = [slice(None)] * x.ndim
    index[axis] = slice(start, stop)
    return x[tuple(index)]


def _patch_pytorch_split_numpy_contract(raw_pytorch, backend, numpy_module) -> None:
    """Patch raw/public PyTorch ``split`` cut indices to follow NumPy semantics."""
    original_split = getattr(raw_pytorch, "split", None)
    if original_split is None:
        return
    if getattr(original_split, "_pyrecest_split_index_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.split = original_split
        return

    def split(x, indices_or_sections, axis=0):
        if not raw_pytorch.is_array(x):
            x = raw_pytorch.array(x)

        cut_indices = numpy_module.asarray(indices_or_sections)
        if isinstance(indices_or_sections, (int, numpy_module.integer)) or cut_indices.ndim == 0:
            return original_split(x, indices_or_sections, axis=axis)
        if cut_indices.ndim != 1:
            raise ValueError("indices_or_sections must be a 1-D sequence")

        bounds = [
            None,
            *(_split_cut_index(index) for index in cut_indices.tolist()),
            None,
        ]
        return tuple(
            _slice_along_axis(x, start, stop, axis)
            for start, stop in zip(bounds, bounds[1:])
        )

    split.__name__ = getattr(original_split, "__name__", "split")
    split.__doc__ = getattr(numpy_module.split, "__doc__", None)
    split._pyrecest_split_index_contract = True
    raw_pytorch.split = split
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.split = split


def patch_pytorch_trapezoid_numpy_contract() -> None:
    """Patch raw/public PyTorch helpers to accept NumPy-style inputs."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    _patch_pytorch_split_numpy_contract(raw_pytorch, backend, np)

    original_trapezoid = getattr(raw_pytorch, "trapezoid", None)
    if original_trapezoid is None:
        return
    if getattr(original_trapezoid, "_pyrecest_numpy_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.trapezoid = original_trapezoid
        return

    def trapezoid(y, x=None, dx=1.0, axis=-1):
        dim = _operator_index(axis)
        device = _preferred_pytorch_device(torch, y, x)
        y = _as_trapezoid_tensor(y, torch, device=device)

        if x is None:
            y = _promote_trapezoid_tensor(y, raw_pytorch)
            return torch.trapezoid(y, dx=dx, dim=dim)

        x = _as_trapezoid_tensor(x, torch, device=y.device)
        result_dtype = torch.promote_types(y.dtype, x.dtype)
        if not (result_dtype.is_floating_point or result_dtype.is_complex):
            result_dtype = raw_pytorch.get_default_dtype()
        y = y.to(dtype=result_dtype)
        x = x.to(dtype=result_dtype)
        return torch.trapezoid(y, x=x, dim=dim)

    trapezoid.__name__ = getattr(original_trapezoid, "__name__", "trapezoid")
    trapezoid.__doc__ = getattr(original_trapezoid, "__doc__", None)
    trapezoid._pyrecest_numpy_contract = True
    raw_pytorch.trapezoid = trapezoid
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.trapezoid = trapezoid
