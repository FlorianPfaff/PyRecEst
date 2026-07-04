"""PyTorch binary-helper device compatibility hook."""

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


def _binary_operands(raw_pytorch, torch_module, left, right):
    """Return operands on a common dtype and an existing preferred device."""
    device = _preferred_pytorch_device(torch_module, left, right)
    left = raw_pytorch.array(left)
    right = raw_pytorch.array(right)
    dtype = torch_module.promote_types(left.dtype, right.dtype)
    if device is None:
        return left.to(dtype=dtype), right.to(dtype=dtype)
    return left.to(device=device, dtype=dtype), right.to(device=device, dtype=dtype)


# Backwards-compatible alias for existing imports/tests that used the old helper name.
_minmax_operands = _binary_operands


def _raw_pytorch_module():
    """Return the raw PyTorch backend module, importing it when available."""
    try:
        return importlib.import_module("pyrecest._backend.pytorch")
    except ModuleNotFoundError:
        return None


def _patch_binary_helpers(raw_pytorch, backend, torch_module, helpers, contract_attr):
    """Patch raw/public binary helpers to share dtype and preserve device."""
    if all(
        getattr(
            getattr(raw_pytorch, helper_name, None),
            contract_attr,
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

        def binary_helper(left, right, _torch_helper=torch_helper):
            left, right = _binary_operands(raw_pytorch, torch_module, left, right)
            return _torch_helper(left, right)

        binary_helper.__name__ = getattr(original_helper, "__name__", helper_name)
        binary_helper.__doc__ = getattr(original_helper, "__doc__", None)
        setattr(binary_helper, contract_attr, True)
        binary_helper._pyrecest_device_contract = True
        setattr(raw_pytorch, helper_name, binary_helper)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, binary_helper)


def patch_pytorch_minmax_device_contract() -> None:
    """Patch raw/public PyTorch binary helpers to preserve device placement."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    raw_pytorch = _raw_pytorch_module()
    if raw_pytorch is None:  # pragma: no cover - backend import failed earlier
        return

    _patch_binary_helpers(
        raw_pytorch,
        backend,
        torch,
        {
            "maximum": torch.maximum,
            "minimum": torch.minimum,
        },
        "_pyrecest_minmax_device_contract",
    )
    _patch_binary_helpers(
        raw_pytorch,
        backend,
        torch,
        {
            "equal": torch.eq,
            "less_equal": torch.le,
            "logical_and": torch.logical_and,
        },
        "_pyrecest_comparison_device_contract",
    )
