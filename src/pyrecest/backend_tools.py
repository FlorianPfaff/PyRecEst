"""Small helpers for inspecting the process-global PyRecEst backend."""

from __future__ import annotations

import os
import warnings
from operator import index as _operator_index


def _pytorch_scalar_tensor_index(index, torch_module):
    """Return Python int indices for scalar integer tensors."""

    if not torch_module.is_tensor(index) or index.ndim != 0:
        return index
    if (
        index.dtype in {torch_module.bool, torch_module.uint8}
        or index.dtype.is_floating_point
        or index.dtype.is_complex
    ):
        return index
    return _operator_index(index)


def _wrap_pytorch_assignment_helper(original_assignment, torch_module):
    """Normalize scalar tensor indices before assignment helper len() checks."""

    def assignment(x, values, indices, axis=0):
        indices = _pytorch_scalar_tensor_index(indices, torch_module)
        return original_assignment(x, values, indices, axis=axis)

    assignment.__name__ = getattr(original_assignment, "__name__", "assignment")
    assignment.__doc__ = getattr(original_assignment, "__doc__", None)
    return assignment


def _patch_pytorch_assignment_scalar_tensor_indices() -> None:
    """Make PyTorch assignment helpers accept scalar integer tensor indices."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    backend.assignment = _wrap_pytorch_assignment_helper(backend.assignment, _torch)
    backend.assignment_by_sum = _wrap_pytorch_assignment_helper(
        backend.assignment_by_sum, _torch
    )
    pytorch_backend.assignment = _wrap_pytorch_assignment_helper(
        pytorch_backend.assignment, _torch
    )
    pytorch_backend.assignment_by_sum = _wrap_pytorch_assignment_helper(
        pytorch_backend.assignment_by_sum, _torch
    )


def _patch_pytorch_diag_numpy_contract() -> None:
    """Make PyTorch diag accept array-like inputs and NumPy's ``k`` keyword."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    if getattr(pytorch_backend.diag, "_pyrecest_numpy_contract", False):
        return

    def diag(v, k=0):
        return _torch.diag(pytorch_backend.array(v), diagonal=k)

    diag.__name__ = getattr(_torch.diag, "__name__", "diag")
    diag.__doc__ = getattr(_torch.diag, "__doc__", None)
    diag._pyrecest_numpy_contract = True
    backend.diag = diag
    pytorch_backend.diag = diag


def _patch_pytorch_broadcast_arrays_numpy_contract() -> None:
    """Make PyTorch broadcast_arrays accept NumPy-style array-like inputs."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    if getattr(pytorch_backend.broadcast_arrays, "_pyrecest_numpy_contract", False):
        return

    def broadcast_arrays(*arrays):
        tensors = tuple(pytorch_backend.array(array) for array in arrays)
        return _torch.broadcast_tensors(*tensors)

    broadcast_arrays.__name__ = "broadcast_arrays"
    broadcast_arrays.__doc__ = getattr(_torch.broadcast_tensors, "__doc__", None)
    broadcast_arrays._pyrecest_numpy_contract = True
    backend.broadcast_arrays = broadcast_arrays
    pytorch_backend.broadcast_arrays = broadcast_arrays


def _patch_pytorch_round_numpy_contract() -> None:
    """Make PyTorch round accept NumPy-style array-like inputs."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    if getattr(pytorch_backend.round, "_pyrecest_numpy_contract", False):
        return

    def round(a, decimals=0, out=None):  # pylint: disable=redefined-builtin
        decimals = _operator_index(decimals)
        result = _torch.round(pytorch_backend.array(a), decimals=decimals)
        if out is not None:
            out.copy_(result)
            return out
        return result

    round.__name__ = getattr(_torch.round, "__name__", "round")
    round.__doc__ = getattr(_torch.round, "__doc__", None)
    round._pyrecest_numpy_contract = True
    backend.round = round
    pytorch_backend.round = round


def _patch_pytorch_special_numpy_contract() -> None:
    """Make PyTorch special functions accept NumPy-style array-like inputs."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    def _return_or_store_out(result, out):
        if out is not None:
            out.copy_(result)
            return out
        return result

    def erf(a, out=None):
        result = _torch.erf(pytorch_backend.array(a))
        return _return_or_store_out(result, out)

    def gammaln(a, out=None):
        result = _torch.special.gammaln(pytorch_backend.array(a))
        return _return_or_store_out(result, out)

    def gamma(a, out=None):
        result = _torch.exp(gammaln(a))
        return _return_or_store_out(result, out)

    def polygamma(n, a, out=None):
        result = _torch.polygamma(n, pytorch_backend.array(a))
        return _return_or_store_out(result, out)

    for name, helper, target in (
        ("erf", erf, _torch.erf),
        ("gammaln", gammaln, _torch.special.gammaln),
        ("gamma", gamma, pytorch_backend.gamma),
        ("polygamma", polygamma, _torch.polygamma),
    ):
        helper.__name__ = name
        helper.__doc__ = getattr(target, "__doc__", None)
        helper._pyrecest_numpy_contract = True
        setattr(backend, name, helper)
        setattr(pytorch_backend, name, helper)

    pytorch_backend._gammaln = gammaln  # pylint: disable=protected-access


def _patch_pytorch_stack_helpers_numpy_contract() -> None:
    """Make PyTorch stack helpers accept NumPy-style array-like inputs."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import numpy as _np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    def _tensor_sequence(tup):
        return [pytorch_backend.array(item) for item in tup]

    def hstack(tup):
        tensors = [_torch.atleast_1d(tensor) for tensor in _tensor_sequence(tup)]
        if not tensors:
            return _torch.cat(tensors, dim=0)
        return _torch.cat(tensors, dim=0 if tensors[0].ndim == 1 else 1)

    def vstack(tup):
        tensors = [_torch.atleast_2d(tensor) for tensor in _tensor_sequence(tup)]
        return _torch.cat(tensors, dim=0)

    def column_stack(tup):
        tensors = []
        for tensor in _tensor_sequence(tup):
            if tensor.ndim < 2:
                tensor = tensor.reshape(-1, 1)
            tensors.append(tensor)
        return _torch.cat(tensors, dim=1)

    def dstack(tup):
        tensors = [_torch.atleast_3d(tensor) for tensor in _tensor_sequence(tup)]
        return _torch.cat(tensors, dim=2)

    for helper_name, helper in {
        "hstack": hstack,
        "vstack": vstack,
        "column_stack": column_stack,
        "dstack": dstack,
    }.items():
        helper.__name__ = helper_name
        helper.__doc__ = getattr(_np, helper_name).__doc__
        helper._pyrecest_numpy_contract = True
        setattr(backend, helper_name, helper)
        setattr(pytorch_backend, helper_name, helper)


def _patch_raw_pytorch_comparison_numpy_contract() -> None:
    """Make raw PyTorch comparison helpers accept NumPy-style array-like inputs."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    def _coerce_binary_args(x, y):
        device = next(
            (value.device for value in (x, y) if _torch.is_tensor(value)),
            None,
        )
        if not _torch.is_tensor(x):
            x = _torch.as_tensor(x, device=device)
        elif device is not None and x.device != device:
            x = x.to(device=device)
        if not _torch.is_tensor(y):
            y = _torch.as_tensor(y, device=device)
        elif device is not None and y.device != device:
            y = y.to(device=device)
        return x, y

    def _wrap_comparison(helper_name, torch_func):
        def comparison(x, y, **kwargs):
            x, y = _coerce_binary_args(x, y)
            return torch_func(x, y, **kwargs)

        comparison.__name__ = getattr(torch_func, "__name__", helper_name)
        comparison.__doc__ = getattr(torch_func, "__doc__", None)
        comparison._pyrecest_numpy_contract = True
        return comparison

    for helper_name, torch_func in (
        ("greater", _torch.greater),
        ("less", _torch.less),
        ("logical_or", _torch.logical_or),
    ):
        helper = _wrap_comparison(helper_name, torch_func)
        setattr(pytorch_backend, helper_name, helper)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, helper)


_patch_pytorch_assignment_scalar_tensor_indices()
_patch_pytorch_diag_numpy_contract()
_patch_pytorch_broadcast_arrays_numpy_contract()
_patch_pytorch_round_numpy_contract()
_patch_pytorch_special_numpy_contract()
_patch_pytorch_stack_helpers_numpy_contract()
_patch_raw_pytorch_comparison_numpy_contract()


def get_backend_name() -> str:
    """Return the backend selected at import time."""
    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    return backend.__backend_name__  # pylint: disable=no-member


def is_backend(expected: str | tuple[str, ...]) -> bool:
    """Return whether the active backend matches one of the expected names."""
    expected_names = _normalize_expected_backend_names(expected)
    return get_backend_name() in expected_names


def _normalize_expected_backend_names(
    expected: str | tuple[str, ...],
) -> tuple[str, ...]:
    message = "expected must name at least one backend."
    if isinstance(expected, str):
        names = (expected,)
    else:
        try:
            names = tuple(expected)
        except TypeError as exc:
            raise ValueError(message) from exc
    if not names or any(
        not isinstance(name, str) or not name or name.strip() != name for name in names
    ):
        raise ValueError(message)
    return tuple(dict.fromkeys(names))


def assert_backend(expected: str | tuple[str, ...]) -> None:
    """Raise ``RuntimeError`` unless the active backend matches ``expected``.

    Parameters
    ----------
    expected : str or tuple[str, ...]
        Allowed backend name or names.
    """
    active = get_backend_name()
    expected_names = _normalize_expected_backend_names(expected)
    if active not in expected_names:
        allowed = ", ".join(expected_names)
        raise RuntimeError(
            f"Expected PyRecEst backend {allowed}; active backend is {active}."
        )


def warn_if_backend_env_changed() -> None:
    """Warn when ``PYRECEST_BACKEND`` no longer matches the imported backend.

    Backend selection is process-global and import-time only. Changing the
    environment variable after importing :mod:`pyrecest` does not switch the
    already constructed backend facade.
    """
    active = get_backend_name()
    requested = os.environ.get("PYRECEST_BACKEND", active)
    if requested != active:
        warnings.warn(
            "PYRECEST_BACKEND was changed after pyrecest was imported. "
            f"The active backend remains {active!r}; the environment now requests "
            f"{requested!r}. Start a new Python process to switch backends.",
            RuntimeWarning,
            stacklevel=2,
        )
