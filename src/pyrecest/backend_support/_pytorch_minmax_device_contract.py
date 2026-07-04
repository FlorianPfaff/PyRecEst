"""PyTorch ``maximum``/``minimum`` device compatibility hook."""

from __future__ import annotations

import importlib


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


def _minmax_operands(raw_pytorch, torch_module, left, right):
    """Return operands on a common dtype and an existing preferred device."""
    device = _preferred_pytorch_device(torch_module, left, right)
    left = raw_pytorch.array(left)
    right = raw_pytorch.array(right)
    dtype = torch_module.promote_types(left.dtype, right.dtype)
    if device is None:
        return left.to(dtype=dtype), right.to(dtype=dtype)
    return left.to(device=device, dtype=dtype), right.to(device=device, dtype=dtype)


def _raw_pytorch_module():
    """Return the raw PyTorch backend module, importing it when available."""
    try:
        return importlib.import_module("pyrecest._backend.pytorch")
    except ModuleNotFoundError:
        return None


def patch_pytorch_minmax_device_contract() -> None:
    """Patch raw/public PyTorch ``maximum`` and ``minimum`` to preserve device."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    raw_pytorch = _raw_pytorch_module()
    if raw_pytorch is None:  # pragma: no cover - backend import failed earlier
        return

    helpers = {
        "maximum": torch.maximum,
        "minimum": torch.minimum,
    }
    if all(
        getattr(
            getattr(raw_pytorch, helper_name, None),
            "_pyrecest_minmax_device_contract",
            False,
        )
        for helper_name in helpers
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for helper_name in helpers:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    for helper_name, torch_helper in helpers.items():
        original_helper = getattr(raw_pytorch, helper_name)

        def minmax(left, right, _torch_helper=torch_helper):
            left, right = _minmax_operands(raw_pytorch, torch, left, right)
            return _torch_helper(left, right)

        minmax.__name__ = getattr(original_helper, "__name__", helper_name)
        minmax.__doc__ = getattr(original_helper, "__doc__", None)
        minmax._pyrecest_minmax_device_contract = True
        minmax._pyrecest_device_contract = True
        setattr(raw_pytorch, helper_name, minmax)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, minmax)
