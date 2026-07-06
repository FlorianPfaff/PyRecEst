"""PyTorch ``dot``/``outer``/``cross`` device compatibility hook."""

from __future__ import annotations

from pyrecest.backend_support._pytorch_sort_numpy_contract import (
    sort_axis_none as _sort_axis_none,
)


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


def _patch_sort_axis_none_contract(raw_pytorch, backend, torch_module) -> None:
    """Patch raw/public PyTorch sort to flatten inputs for ``axis=None``."""
    original_sort = getattr(raw_pytorch, "sort", None)
    if original_sort is None:
        return
    if getattr(original_sort, "_pyrecest_sort_axis_none_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, "sort", original_sort)
        return

    def sort(a, axis=-1, kind=None, order=None, *, stable=None, descending=False):
        return _sort_axis_none(
            raw_pytorch,
            torch_module,
            a,
            axis=axis,
            kind=kind,
            order=order,
            stable=stable,
            descending=descending,
        )

    sort.__name__ = getattr(original_sort, "__name__", "sort")
    sort.__doc__ = getattr(original_sort, "__doc__", None)
    sort._pyrecest_sort_axis_none_contract = True
    sort._pyrecest_numpy_contract = True
    setattr(raw_pytorch, "sort", sort)
    if getattr(backend, "__backend_name__", None) == "pytorch":
        setattr(backend, "sort", sort)


def patch_pytorch_dot_outer_device_contract() -> None:
    """Patch raw/public PyTorch vector-product helpers to preserve non-CPU operands."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    _patch_sort_axis_none_contract(raw_pytorch, backend, torch)

    original_dot = getattr(raw_pytorch, "dot", None)
    original_outer = getattr(raw_pytorch, "outer", None)
    original_cross = getattr(raw_pytorch, "cross", None)
    if original_dot is None or original_outer is None:
        return

    dot_outer_patched = getattr(
        original_dot,
        "_pyrecest_dot_outer_device_contract",
        False,
    ) and getattr(
        original_outer,
        "_pyrecest_dot_outer_device_contract",
        False,
    )
    cross_patched = original_cross is None or getattr(
        original_cross,
        "_pyrecest_cross_device_contract",
        False,
    )
    if dot_outer_patched and cross_patched:
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.dot = original_dot
            backend.outer = original_outer
            if original_cross is not None:
                backend.cross = original_cross
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

    if original_cross is None:
        return

    def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        a, b = _promoted_pair(raw_pytorch, torch, a, b)
        return original_cross(
            a,
            b,
            axisa=axisa,
            axisb=axisb,
            axisc=axisc,
            axis=axis,
        )

    cross.__name__ = getattr(original_cross, "__name__", "cross")
    cross.__doc__ = getattr(original_cross, "__doc__", None)
    cross._pyrecest_cross_contract = getattr(
        original_cross,
        "_pyrecest_cross_contract",
        True,
    )
    cross._pyrecest_cross_device_contract = True
    cross._pyrecest_device_contract = True
    cross._pyrecest_numpy_contract = getattr(
        original_cross,
        "_pyrecest_numpy_contract",
        True,
    )
    setattr(raw_pytorch, "cross", cross)
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.cross = cross
