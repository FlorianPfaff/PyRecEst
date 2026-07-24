"""PyTorch ``where`` device compatibility hook."""

from __future__ import annotations

import importlib

from pyrecest.backend_support._pytorch_array_equal_mixed_dtype_contract import (
    patch_pytorch_array_equal_mixed_dtype_contract as _patch_pytorch_array_equal_mixed_dtype_contract,
)


def _preferred_pytorch_device(torch_module, *values):
    """Return a non-CPU tensor device when mixed-device operands are present."""
    for value in values:
        if torch_module.is_tensor(value) and value.device.type == "meta":
            return value.device
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def _tensor_on_device(torch_module, value, *, device, dtype=None):
    """Return ``value`` as a tensor on ``device`` without losing the preferred device."""
    if torch_module.is_tensor(value):
        target_dtype = dtype if dtype is not None else value.dtype
        if device is None and target_dtype == value.dtype:
            return value
        return value.to(
            device=device if device is not None else value.device,
            dtype=target_dtype,
        )
    return torch_module.as_tensor(value, dtype=dtype, device=device)


def _raw_pytorch_module():
    """Return the raw PyTorch backend module, importing it when available."""
    try:
        return importlib.import_module("pyrecest._backend.pytorch")
    except ModuleNotFoundError:
        return None


def patch_pytorch_where_device_contract() -> None:
    """Patch raw/public PyTorch ``where`` to preserve an existing non-CPU device."""
    _patch_pytorch_array_equal_mixed_dtype_contract()

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    raw_pytorch = _raw_pytorch_module()
    if raw_pytorch is None:  # pragma: no cover - backend import failed earlier
        return

    original_where = getattr(raw_pytorch, "where", None)
    if original_where is None:
        return
    if getattr(original_where, "_pyrecest_where_device_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.where = original_where
        return

    def where(condition, x=None, y=None):
        if (x is None) != (y is None):
            raise ValueError("either both or neither of x and y should be given")

        device = _preferred_pytorch_device(torch, condition, x, y)
        condition = _tensor_on_device(
            torch,
            condition,
            device=device,
            dtype=torch.bool,
        )
        if x is None and y is None:
            return torch.where(condition)

        x = _tensor_on_device(torch, x, device=device)
        y = _tensor_on_device(torch, y, device=device)
        result_dtype = torch.result_type(x, y)
        x = x.to(dtype=result_dtype)
        y = y.to(dtype=result_dtype)
        return torch.where(condition, x, y)

    where.__name__ = getattr(original_where, "__name__", "where")
    where.__doc__ = getattr(original_where, "__doc__", None)
    where._pyrecest_where_device_contract = True
    where._pyrecest_device_contract = True
    raw_pytorch.where = where
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.where = where
