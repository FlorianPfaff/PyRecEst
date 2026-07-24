"""PyTorch ``array_equal`` mixed-dtype compatibility hook."""

from __future__ import annotations


def _preferred_pytorch_device(torch_module, *values):
    """Return a non-CPU tensor device when mixed-device operands are present."""
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def _tensor_on_device(torch_module, value, *, device):
    """Return ``value`` as a tensor on the preferred existing device."""
    if torch_module.is_tensor(value):
        if device is not None and value.device != device:
            return value.to(device=device)
        return value
    return torch_module.as_tensor(value, device=device)


def _is_integral_dtype(dtype) -> bool:
    """Return whether a PyTorch dtype is integral or boolean."""
    return not dtype.is_floating_point and not dtype.is_complex


def _to_numpy_exact(value, torch_module):
    """Convert a dense tensor to NumPy without losing bfloat16 values."""
    if value.dtype == torch_module.bfloat16:
        value = value.to(dtype=torch_module.float32)
    return value.detach().cpu().numpy()


def patch_pytorch_array_equal_mixed_dtype_contract() -> None:
    """Prevent lossy or unsupported promotion in PyTorch ``array_equal``."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_array_equal = getattr(raw_pytorch, "array_equal", None)
    if original_array_equal is None or not getattr(
        original_array_equal,
        "_pyrecest_equal_nan_contract",
        False,
    ):
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    if getattr(
        original_array_equal,
        "_pyrecest_exact_mixed_dtype_contract",
        False,
    ):
        if active_pytorch_backend:
            backend.array_equal = original_array_equal
        return

    def array_equal(a, b, equal_nan=False):
        device = _preferred_pytorch_device(torch, a, b)
        a = _tensor_on_device(torch, a, device=device)
        b = _tensor_on_device(torch, b, device=device)
        if tuple(a.shape) != tuple(b.shape):
            return False

        if a.dtype != b.dtype and (
            _is_integral_dtype(a.dtype) or _is_integral_dtype(b.dtype)
        ):
            return bool(
                np.array_equal(
                    _to_numpy_exact(a, torch),
                    _to_numpy_exact(b, torch),
                    equal_nan=equal_nan,
                )
            )
        return original_array_equal(a, b, equal_nan=equal_nan)

    array_equal.__name__ = getattr(original_array_equal, "__name__", "array_equal")
    array_equal.__doc__ = getattr(original_array_equal, "__doc__", None)
    array_equal._pyrecest_equal_nan_contract = True
    array_equal._pyrecest_numpy_contract = True
    array_equal._pyrecest_exact_mixed_dtype_contract = True
    raw_pytorch.array_equal = array_equal
    if active_pytorch_backend:
        backend.array_equal = array_equal


__all__ = ["patch_pytorch_array_equal_mixed_dtype_contract"]
