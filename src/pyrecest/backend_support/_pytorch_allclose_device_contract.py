"""PyTorch compatibility hooks used during stability initialization."""

from __future__ import annotations

from operator import index as _operator_index

from pyrecest.backend_support._pytorch_creation_shape_contract import (
    patch_pytorch_creation_shape_contract as _patch_pytorch_creation_shape_contract,
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


def _coerce_binary_args(torch_module, x, y):
    """Move array-like PyTorch binary operands to a preferred existing device."""

    device = _preferred_pytorch_device(torch_module, x, y)
    if not torch_module.is_tensor(x):
        x = torch_module.as_tensor(x, device=device)
    elif device is not None and x.device != device:
        x = x.to(device=device)
    if not torch_module.is_tensor(y):
        y = torch_module.as_tensor(y, device=device)
    elif device is not None and y.device != device:
        y = y.to(device=device)
    return x, y


def _coerce_array_to_device(pytorch_backend, value, *, device):
    """Return one backend tensor on the preferred existing tensor device."""

    tensor = pytorch_backend.array(value)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device=device)
    return tensor


def _patch_pytorch_linalg_logm_arraylike_contract() -> None:
    """Patch raw/public PyTorch ``linalg.logm`` to normalize array-like inputs."""

    try:
        import pyrecest._backend.pytorch.linalg as pytorch_linalg  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_logm = getattr(pytorch_linalg, "logm", None)
    if original_logm is None:
        return
    if getattr(original_logm, "_pyrecest_arraylike_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.linalg.logm = original_logm
        return

    def logm(x):
        return original_logm(
            pytorch_linalg._as_linalg_tensor(x)
        )  # pylint: disable=protected-access

    logm.__name__ = getattr(original_logm, "__name__", "logm")
    logm.__doc__ = getattr(original_logm, "__doc__", None)
    logm._pyrecest_arraylike_contract = True
    pytorch_linalg.logm = logm
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.linalg.logm = logm


def _patch_pytorch_flip_numpy_axis_contract() -> None:
    """Patch raw/public PyTorch ``flip`` to accept NumPy integer axes."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_flip = getattr(pytorch_backend, "flip", None)
    if original_flip is None:
        return
    if getattr(original_flip, "_pyrecest_numpy_axis_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.flip = original_flip
        return

    def _flip_axes(axis, ndim):
        if axis is None:
            return list(range(ndim))
        if isinstance(axis, (int, np.integer)):
            return [int(axis)]
        return [int(_operator_index(one_axis)) for one_axis in axis]

    def flip(x, axis):
        x = pytorch_backend.array(x)
        return torch_module.flip(x, dims=_flip_axes(axis, x.ndim))

    flip.__name__ = getattr(original_flip, "__name__", "flip")
    flip.__doc__ = getattr(original_flip, "__doc__", None)
    flip._pyrecest_numpy_axis_contract = True
    pytorch_backend.flip = flip
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.flip = flip


def _patch_apply_along_axis_arguments(pytorch_backend, backend) -> None:
    """Patch PyTorch ``apply_along_axis`` to forward callback arguments."""

    original_apply_along_axis = getattr(pytorch_backend, "apply_along_axis", None)
    if original_apply_along_axis is None:
        return
    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    if getattr(
        original_apply_along_axis,
        "_pyrecest_argument_forwarding_contract",
        False,
    ):
        if active_pytorch_backend:
            backend.apply_along_axis = original_apply_along_axis
        return

    def apply_along_axis(func, axis, tensor, *args, **kwargs):
        def wrapped_func(tensor_slice):
            return func(tensor_slice, *args, **kwargs)

        return original_apply_along_axis(wrapped_func, axis, tensor)

    apply_along_axis.__name__ = getattr(
        original_apply_along_axis,
        "__name__",
        "apply_along_axis",
    )
    apply_along_axis.__doc__ = getattr(original_apply_along_axis, "__doc__", None)
    apply_along_axis._pyrecest_argument_forwarding_contract = True
    pytorch_backend.apply_along_axis = apply_along_axis
    if active_pytorch_backend:
        backend.apply_along_axis = apply_along_axis


def _close_operands(pytorch_backend, torch_module, a, b):
    """Return close-comparison operands on a common device and dtype."""

    a, b = _coerce_binary_args(torch_module, a, b)
    dtype = torch_module.promote_types(a.dtype, b.dtype)
    a = a.to(dtype=dtype)
    b = b.to(dtype=dtype)
    return torch_module.broadcast_tensors(a, b)


def _mark_close_contract(helper):
    """Mark a patched close helper as satisfying both device and NaN contracts."""

    helper._pyrecest_device_contract = True
    helper._pyrecest_equal_nan_device_contract = True
    helper._pyrecest_missing_value_contract = True
    return helper


def _patch_allclose(pytorch_backend, backend, torch_module) -> None:
    """Patch raw/public PyTorch ``allclose`` to preserve non-CPU operands."""

    original_allclose = getattr(pytorch_backend, "allclose", None)
    if original_allclose is None:
        return
    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    if getattr(original_allclose, "_pyrecest_equal_nan_device_contract", False):
        if active_pytorch_backend:
            backend.allclose = original_allclose
        return

    def allclose(
        a,
        b,
        atol=pytorch_backend.atol,
        rtol=pytorch_backend.rtol,
        equal_nan=False,
    ):
        a, b = _close_operands(pytorch_backend, torch_module, a, b)
        return torch_module.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    allclose.__name__ = getattr(original_allclose, "__name__", "allclose")
    allclose.__doc__ = getattr(original_allclose, "__doc__", None)
    _mark_close_contract(allclose)
    pytorch_backend.allclose = allclose
    if active_pytorch_backend:
        backend.allclose = allclose


def _patch_isclose(pytorch_backend, backend, torch_module) -> None:
    """Patch raw/public PyTorch ``isclose`` to preserve non-CPU operands."""

    original_isclose = getattr(pytorch_backend, "isclose", None)
    if original_isclose is None:
        return
    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    if getattr(original_isclose, "_pyrecest_equal_nan_device_contract", False):
        if active_pytorch_backend:
            backend.isclose = original_isclose
        return

    def isclose(
        a,
        b,
        rtol=pytorch_backend.rtol,
        atol=pytorch_backend.atol,
        equal_nan=False,
    ):
        a, b = _close_operands(pytorch_backend, torch_module, a, b)
        return torch_module.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    isclose.__name__ = getattr(original_isclose, "__name__", "isclose")
    isclose.__doc__ = getattr(original_isclose, "__doc__", None)
    _mark_close_contract(isclose)
    pytorch_backend.isclose = isclose
    if active_pytorch_backend:
        backend.isclose = isclose


def _patch_broadcast_arrays(pytorch_backend, backend, torch_module) -> None:
    """Patch raw/public PyTorch ``broadcast_arrays`` to preserve tensor devices."""

    original_broadcast_arrays = getattr(pytorch_backend, "broadcast_arrays", None)
    if original_broadcast_arrays is None:
        return
    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    if getattr(
        original_broadcast_arrays,
        "_pyrecest_broadcast_arrays_device_contract",
        False,
    ):
        if active_pytorch_backend:
            backend.broadcast_arrays = original_broadcast_arrays
        return

    def broadcast_arrays(*arrays):
        device = _preferred_pytorch_device(torch_module, *arrays)
        tensors = tuple(
            _coerce_array_to_device(
                pytorch_backend,
                value,
                device=device,
            )
            for value in arrays
        )
        return original_broadcast_arrays(*tensors)

    broadcast_arrays.__name__ = getattr(
        original_broadcast_arrays,
        "__name__",
        "broadcast_arrays",
    )
    broadcast_arrays.__doc__ = getattr(original_broadcast_arrays, "__doc__", None)
    broadcast_arrays._pyrecest_numpy_contract = True
    broadcast_arrays._pyrecest_broadcast_arrays_device_contract = True
    broadcast_arrays._pyrecest_device_contract = True
    pytorch_backend.broadcast_arrays = broadcast_arrays
    if active_pytorch_backend:
        backend.broadcast_arrays = broadcast_arrays


def patch_pytorch_allclose_device_contract() -> None:
    """Patch raw/public PyTorch close helpers to preserve existing tensor devices."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    _patch_pytorch_creation_shape_contract()
    _patch_pytorch_linalg_logm_arraylike_contract()
    _patch_pytorch_flip_numpy_axis_contract()
    _patch_apply_along_axis_arguments(pytorch_backend, backend)
    _patch_allclose(pytorch_backend, backend, torch_module)
    _patch_isclose(pytorch_backend, backend, torch_module)
    _patch_broadcast_arrays(pytorch_backend, backend, torch_module)
