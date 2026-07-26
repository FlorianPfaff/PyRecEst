"""PyTorch stack-helper device compatibility hook."""

from __future__ import annotations

import importlib

_STACK_HELPER_NAMES = (
    "concatenate",
    "stack",
    "hstack",
    "vstack",
    "column_stack",
    "dstack",
)
_EMPTY_CONCATENATE_MESSAGE = "need at least one array to concatenate"
_EMPTY_STACK_MESSAGE = "need at least one array to stack"


def _raw_pytorch_module():
    """Return the raw PyTorch backend module, importing it when available."""
    try:
        return importlib.import_module("pyrecest._backend.pytorch")
    except ModuleNotFoundError:
        return None


def _preferred_pytorch_device(torch_module, *values):
    """Return an existing tensor device, preferring non-copyable meta tensors."""
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


def _tensor_sequence(raw_pytorch, torch_module, values):
    """Return sequence entries normalized onto one preferred tensor device."""
    values = tuple(values)
    device = _preferred_pytorch_device(torch_module, *values)
    tensors = []
    for value in values:
        tensor = raw_pytorch.array(value)
        if device is not None and tensor.device != device:
            tensor = tensor.to(device=device)
        tensors.append(tensor)
    if not tensors:
        return tensors
    return raw_pytorch.convert_to_wider_dtype(tensors)


def _require_nonempty_stack_sequence(tensors, message=_EMPTY_CONCATENATE_MESSAGE):
    """Raise the NumPy stack-helper error for empty input sequences."""
    if not tensors:
        raise ValueError(message)


def _build_stack_helpers(raw_pytorch, torch_module):
    """Build NumPy-style PyTorch stack helpers with consistent devices."""

    def concatenate(tup, axis=0, out=None):
        tensors = _tensor_sequence(raw_pytorch, torch_module, tup)
        _require_nonempty_stack_sequence(tensors)
        if axis is None:
            tensors = [tensor.reshape(-1) for tensor in tensors]
            axis = 0
        return torch_module.cat(tensors, dim=axis, out=out)

    def stack(seq, axis=0, out=None, *, dim=None):
        if dim is not None:
            if axis not in (0, dim):
                raise TypeError("stack() got both 'axis' and 'dim'")
            axis = dim
        tensors = _tensor_sequence(raw_pytorch, torch_module, seq)
        _require_nonempty_stack_sequence(tensors, _EMPTY_STACK_MESSAGE)
        return torch_module.stack(tensors, dim=axis, out=out)

    def hstack(tup):
        tensors = [
            torch_module.atleast_1d(tensor)
            for tensor in _tensor_sequence(raw_pytorch, torch_module, tup)
        ]
        _require_nonempty_stack_sequence(tensors)
        return torch_module.cat(tensors, dim=0 if tensors[0].ndim == 1 else 1)

    def vstack(tup):
        tensors = [
            torch_module.atleast_2d(tensor)
            for tensor in _tensor_sequence(raw_pytorch, torch_module, tup)
        ]
        _require_nonempty_stack_sequence(tensors)
        return torch_module.cat(tensors, dim=0)

    def column_stack(tup):
        tensors = []
        for tensor in _tensor_sequence(raw_pytorch, torch_module, tup):
            if tensor.ndim < 2:
                tensor = tensor.reshape(-1, 1)
            tensors.append(tensor)
        _require_nonempty_stack_sequence(tensors)
        return torch_module.cat(tensors, dim=1)

    def dstack(tup):
        tensors = [
            torch_module.atleast_3d(tensor)
            for tensor in _tensor_sequence(raw_pytorch, torch_module, tup)
        ]
        _require_nonempty_stack_sequence(tensors)
        return torch_module.cat(tensors, dim=2)

    return {
        "concatenate": concatenate,
        "stack": stack,
        "hstack": hstack,
        "vstack": vstack,
        "column_stack": column_stack,
        "dstack": dstack,
    }


def patch_pytorch_stack_helpers_device_contract() -> None:
    """Patch raw/public PyTorch stack helpers to preserve tensor device."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    raw_pytorch = _raw_pytorch_module()
    if raw_pytorch is None:  # pragma: no cover - backend import failed earlier
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    if all(
        getattr(
            getattr(raw_pytorch, helper_name, None),
            "_pyrecest_stack_helpers_device_contract",
            False,
        )
        for helper_name in _STACK_HELPER_NAMES
    ):
        if active_pytorch_backend:
            for helper_name in _STACK_HELPER_NAMES:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    helpers = _build_stack_helpers(raw_pytorch, torch)
    for helper_name, helper in helpers.items():
        original_helper = getattr(raw_pytorch, helper_name, None)
        if original_helper is None:
            continue
        helper.__name__ = helper_name
        helper.__doc__ = getattr(np, helper_name).__doc__
        helper._pyrecest_numpy_contract = True
        helper._pyrecest_device_contract = True
        helper._pyrecest_stack_helpers_device_contract = True
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)
