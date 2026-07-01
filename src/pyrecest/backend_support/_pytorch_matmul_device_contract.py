"""PyTorch ``matmul`` device compatibility hook."""

from __future__ import annotations


def _preferred_pytorch_device(torch_module, *values):
    """Return an existing non-CPU tensor device, falling back to any tensor."""
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def patch_pytorch_matmul_device_contract() -> None:
    """Patch raw/public PyTorch ``matmul`` to keep operands on one device."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_matmul = getattr(raw_pytorch, "matmul", None)
    if original_matmul is None:
        return
    if getattr(original_matmul, "_pyrecest_device_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.matmul = original_matmul
        return

    def matmul(x, y, out=None):
        device = _preferred_pytorch_device(torch, x, y, out)
        x = raw_pytorch.array(x)
        y = raw_pytorch.array(y)
        dtype = torch.promote_types(x.dtype, y.dtype)

        if device is not None:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
        else:
            x = x.to(dtype=dtype)
            y = y.to(dtype=dtype)

        if out is not None:
            return torch.matmul(x, y, out=out)
        return torch.matmul(x, y)

    matmul.__name__ = getattr(original_matmul, "__name__", "matmul")
    matmul.__doc__ = getattr(original_matmul, "__doc__", None)
    matmul._pyrecest_device_contract = True
    matmul._pyrecest_numpy_contract = True
    raw_pytorch.matmul = matmul
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.matmul = matmul
