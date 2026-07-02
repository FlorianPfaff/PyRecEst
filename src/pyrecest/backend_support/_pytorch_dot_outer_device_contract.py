"""PyTorch ``dot``/``outer`` device compatibility hook."""

from __future__ import annotations


# Keep this hook aligned with the backend dot tests.
def _preferred_pytorch_device(torch_module, *values):
    """Return a non-CPU tensor device when mixed-device operands are present."""
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def _promoted_pair(raw_pytorch, torch_module, left, right):
    """Return PyTorch operands on a common dtype and preferred existing device."""
    device = _preferred_pytorch_device(torch_module, left, right)
    left = raw_pytorch.array(left)
    right = raw_pytorch.array(right)
    dtype = torch_module.promote_types(left.dtype, right.dtype)
    if device is None:
        return left.to(dtype=dtype), right.to(dtype=dtype)
    return left.to(device=device, dtype=dtype), right.to(device=device, dtype=dtype)


def patch_pytorch_dot_outer_device_contract() -> None:
    """Patch raw/public PyTorch ``dot`` and ``outer`` to preserve non-CPU operands."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_dot = getattr(raw_pytorch, "dot", None)
    original_outer = getattr(raw_pytorch, "outer", None)
    if original_dot is None or original_outer is None:
        return
    if getattr(original_dot, "_pyrecest_dot_outer_device_contract", False) and getattr(
        original_outer,
        "_pyrecest_dot_outer_device_contract",
        False,
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.dot = original_dot
            backend.outer = original_outer
        return

    def dot(a, b):
        a, b = _promoted_pair(raw_pytorch, torch, a, b)
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
        a, b = _promoted_pair(raw_pytorch, torch, a, b)
        if a.ndim == 0 or b.ndim == 0:
            return torch.multiply(a, b)
        return a[..., :, None] * b[..., None, :]

    for helper_name, helper, original_helper in (
        ("dot", dot, original_dot),
        ("outer", outer, original_outer),
    ):
        helper.__name__ = getattr(original_helper, "__name__", helper_name)
        helper.__doc__ = getattr(original_helper, "__doc__", None)
        helper._pyrecest_dot_outer_device_contract = True
        helper._pyrecest_device_contract = True
        helper._pyrecest_numpy_contract = True
        setattr(raw_pytorch, helper_name, helper)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, helper)
