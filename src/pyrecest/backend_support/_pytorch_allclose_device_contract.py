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
        return original_logm(pytorch_linalg._as_linalg_tensor(x))  # pylint: disable=protected-access

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


def _patch_pytorch_stack_helpers_arraylike_contract() -> None:
    """Patch raw/public PyTorch stack helpers to accept array-like sequences."""

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    helper_names = ("hstack", "vstack", "column_stack", "dstack")
    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    if all(
        getattr(
            getattr(pytorch_backend, helper_name, None),
            "_pyrecest_arraylike_contract",
            False,
        )
        for helper_name in helper_names
    ):
        if active_pytorch_backend:
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(pytorch_backend, helper_name))
        return

    def _tensor_sequence(seq):
        return [pytorch_backend.array(item) for item in seq]

    def hstack(tup):
        tensors = [torch.atleast_1d(tensor) for tensor in _tensor_sequence(tup)]
        if not tensors:
            return torch.cat(tensors, dim=0)
        return torch.cat(tensors, dim=0 if tensors[0].ndim == 1 else 1)

    def vstack(tup):
        tensors = [torch.atleast_2d(tensor) for tensor in _tensor_sequence(tup)]
        return torch.cat(tensors, dim=0)

    def column_stack(tup):
        tensors = []
        for tensor in _tensor_sequence(tup):
            if tensor.ndim < 2:
                tensor = tensor.reshape(-1, 1)
            tensors.append(tensor)
        return torch.cat(tensors, dim=1)

    def dstack(tup):
        tensors = [torch.atleast_3d(tensor) for tensor in _tensor_sequence(tup)]
        return torch.cat(tensors, dim=2)

    helpers = {
        "hstack": hstack,
        "vstack": vstack,
        "column_stack": column_stack,
        "dstack": dstack,
    }
    for helper_name, helper in helpers.items():
        helper.__name__ = helper_name
        helper.__doc__ = getattr(np, helper_name).__doc__
        helper._pyrecest_arraylike_contract = True
        setattr(pytorch_backend, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


def patch_pytorch_allclose_device_contract() -> None:
    """Patch raw/public PyTorch ``allclose`` to preserve non-CPU operands."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    _patch_pytorch_creation_shape_contract()
    _patch_pytorch_linalg_logm_arraylike_contract()
    _patch_pytorch_flip_numpy_axis_contract()
    _patch_pytorch_stack_helpers_arraylike_contract()

    original_allclose = getattr(pytorch_backend, "allclose", None)
    if original_allclose is None:
        return
    if getattr(original_allclose, "_pyrecest_equal_nan_device_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.allclose = original_allclose
        return

    def allclose(
        a,
        b,
        atol=pytorch_backend.atol,
        rtol=pytorch_backend.rtol,
        equal_nan=False,
    ):
        a, b = _coerce_binary_args(torch_module, a, b)
        a, b = pytorch_backend.convert_to_wider_dtype([a, b])
        a, b = torch_module.broadcast_tensors(a, b)
        return torch_module.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    allclose.__name__ = getattr(original_allclose, "__name__", "allclose")
    allclose.__doc__ = getattr(original_allclose, "__doc__", None)
    allclose._pyrecest_equal_nan_device_contract = True
    pytorch_backend.allclose = allclose
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.allclose = allclose
