"""Backend capability metadata with PyTorch dot/outer device alignment."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_base_capabilities_module():
    module_path = Path(__file__).resolve().parent.parent / "capabilities.py"
    spec = importlib.util.spec_from_file_location(
        "_pyrecest_backend_capabilities_base",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load backend capabilities module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_CAPABILITIES = _load_base_capabilities_module()

for _name in dir(_BASE_CAPABILITIES):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_BASE_CAPABILITIES, _name)


def _preferred_pytorch_device(torch_module, *values):
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def _as_pytorch_tensor_on_device(value, torch_module, *, device, dtype=None):
    if torch_module.is_tensor(value):
        if device is not None and value.device != device:
            value = value.to(device=device)
        if dtype is not None and value.dtype != dtype:
            value = value.to(dtype=dtype)
        return value
    return torch_module.as_tensor(value, dtype=dtype, device=device)


def _linear_operands(a, b, raw_pytorch, torch_module):
    device = _preferred_pytorch_device(torch_module, a, b)
    a = raw_pytorch.array(a)
    b = raw_pytorch.array(b)
    dtype = torch_module.promote_types(a.dtype, b.dtype)
    return (
        _as_pytorch_tensor_on_device(a, torch_module, device=device, dtype=dtype),
        _as_pytorch_tensor_on_device(b, torch_module, device=device, dtype=dtype),
    )


def _patch_pytorch_dot_outer_device_contract() -> None:
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    helper_names = ("dot", "outer")
    if all(
        getattr(
            getattr(raw_pytorch, helper_name, None), "_pyrecest_device_contract", False
        )
        for helper_name in helper_names
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    original_dot = raw_pytorch.dot
    original_outer = raw_pytorch.outer

    def dot(a, b):
        a, b = _linear_operands(a, b, raw_pytorch, torch)
        if a.ndim == 0 or b.ndim == 0:
            return torch.multiply(a, b)
        if a.ndim == 1 and b.ndim == 1:
            return torch.dot(a, b)
        if b.ndim == 1:
            return torch.einsum("...i,i->...", a, b)
        if a.ndim == 1:
            return torch.einsum("i,...i->...", a, b)
        return torch.einsum("...i,...i->...", a, b)

    def outer(a, b):
        a, b = _linear_operands(a, b, raw_pytorch, torch)
        if a.ndim == 0 or b.ndim == 0:
            return torch.multiply(a, b)
        return a[..., :, None] * b[..., None, :]

    dot.__name__ = getattr(original_dot, "__name__", "dot")
    dot.__doc__ = getattr(original_dot, "__doc__", None)
    dot._pyrecest_numpy_contract = True
    dot._pyrecest_device_contract = True
    outer.__name__ = getattr(original_outer, "__name__", "outer")
    outer.__doc__ = getattr(original_outer, "__doc__", None)
    outer._pyrecest_numpy_contract = True
    outer._pyrecest_device_contract = True

    raw_pytorch.dot = dot
    raw_pytorch.outer = outer
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.dot = dot
        backend.outer = outer


_patch_pytorch_dot_outer_device_contract()


def __getattr__(name):
    return getattr(_BASE_CAPABILITIES, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_BASE_CAPABILITIES)))


__all__ = getattr(
    _BASE_CAPABILITIES,
    "__all__",
    [name for name in dir(_BASE_CAPABILITIES) if not name.startswith("_")],
)
