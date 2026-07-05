"""PyTorch ``trapezoid`` NumPy compatibility hook."""

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


def _trapezoid_axis(axis) -> int:
    """Return a NumPy-style integer axis while rejecting boolean axes."""
    if isinstance(axis, bool) or type(axis).__name__ == "bool_":
        raise TypeError("axis must be an integer")
    try:
        return _operator_index(axis)
    except TypeError as exc:
        raise TypeError("axis must be an integer") from exc


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


def patch_pytorch_trapezoid_numpy_contract() -> None:
    """Patch raw/public PyTorch ``trapezoid`` to accept NumPy-style inputs."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_trapezoid = getattr(raw_pytorch, "trapezoid", None)
    if original_trapezoid is None:
        return
    if getattr(original_trapezoid, "_pyrecest_numpy_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.trapezoid = original_trapezoid
        return

    def trapezoid(y, x=None, dx=1.0, axis=-1):
        dim = _trapezoid_axis(axis)
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
