"""Backend capability metadata with PyTorch dot/outer device alignment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, cast


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

BACKEND_CAPABILITIES = _BASE_CAPABILITIES.BACKEND_CAPABILITIES
API_BACKEND_CAPABILITIES = _BASE_CAPABILITIES.API_BACKEND_CAPABILITIES
BACKEND_SUPPORT_LEVELS = _BASE_CAPABILITIES.BACKEND_SUPPORT_LEVELS
REQUIRED_BACKENDS = _BASE_CAPABILITIES.REQUIRED_BACKENDS
_ALLOWED_API_CAPABILITY_KEYS = _BASE_CAPABILITIES._ALLOWED_API_CAPABILITY_KEYS


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


def get_unsupported_functions(
    backend_name: str, module_name: str = ""
) -> tuple[str, ...]:
    """Return unsupported facade functions for a backend module."""
    backend = cast(dict[str, Any], BACKEND_CAPABILITIES.get(backend_name, {}))
    unsupported = cast(dict[str, tuple[str, ...]], backend.get("unsupported", {}))
    return tuple(unsupported.get(module_name, ()))


def get_partial_capabilities(
    backend_name: str, module_name: str = ""
) -> dict[str, str]:
    """Return partial-support notes for a backend module."""
    backend = cast(dict[str, Any], BACKEND_CAPABILITIES.get(backend_name, {}))
    partial = cast(dict[str, dict[str, str]], backend.get("partial", {}))
    return dict(partial.get(module_name, {}))


def get_bridged_capabilities(
    backend_name: str, module_name: str = ""
) -> dict[str, str]:
    """Return operations that work by crossing into another numerical stack."""
    backend = cast(dict[str, Any], BACKEND_CAPABILITIES.get(backend_name, {}))
    bridged = cast(dict[str, dict[str, str]], backend.get("bridged", {}))
    return dict(bridged.get(module_name, {}))


def get_api_backend_support(api_name: str) -> dict[str, str]:
    """Return backend support metadata for a public API name."""
    return dict(API_BACKEND_CAPABILITIES.get(api_name, {}))


def iter_api_backend_capabilities() -> tuple[tuple[str, dict[str, str]], ...]:
    """Return public API backend support rows in a stable order."""
    return tuple(sorted(API_BACKEND_CAPABILITIES.items()))


def validate_api_backend_capabilities() -> tuple[str, ...]:
    """Return human-readable validation errors for API capability metadata."""
    errors: list[str] = []
    for api_name, row in iter_api_backend_capabilities():
        if not api_name:
            errors.append("Capability row has an empty API name.")

        unknown_keys = sorted(set(row) - _ALLOWED_API_CAPABILITY_KEYS)
        if unknown_keys:
            errors.append(
                f"{api_name}: unknown capability entries for {', '.join(unknown_keys)}."
            )

        missing_backends = [
            backend for backend in REQUIRED_BACKENDS if backend not in row
        ]
        if missing_backends:
            errors.append(
                f"{api_name}: missing backend support entries for {', '.join(missing_backends)}."
            )

        for backend_name in REQUIRED_BACKENDS:
            support_level = row.get(backend_name)
            if support_level not in BACKEND_SUPPORT_LEVELS:
                errors.append(
                    f"{api_name}: unsupported support level {support_level!r} for {backend_name}."
                )

        if not row.get("notes"):
            errors.append(f"{api_name}: missing explanatory notes.")

    return tuple(errors)


def __getattr__(name):
    return getattr(_BASE_CAPABILITIES, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_BASE_CAPABILITIES)))


__all__ = getattr(
    _BASE_CAPABILITIES,
    "__all__",
    [name for name in dir(_BASE_CAPABILITIES) if not name.startswith("_")],
)
