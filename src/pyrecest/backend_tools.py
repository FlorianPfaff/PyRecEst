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


def _preferred_pytorch_device(torch_module, *values):
    """Return a non-CPU tensor device when mixed-device operands are present."""

    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


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

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    if getattr(pytorch_backend.broadcast_arrays, "_pyrecest_numpy_contract", False):
        if active_pytorch_backend:
            backend.broadcast_arrays = pytorch_backend.broadcast_arrays
        return

    def broadcast_arrays(*arrays):
        tensors = tuple(pytorch_backend.array(array) for array in arrays)
        return _torch.broadcast_tensors(*tensors)

    broadcast_arrays.__name__ = "broadcast_arrays"
    broadcast_arrays.__doc__ = getattr(_torch.broadcast_tensors, "__doc__", None)
    broadcast_arrays._pyrecest_numpy_contract = True
    pytorch_backend.broadcast_arrays = broadcast_arrays
    if active_pytorch_backend:
        backend.broadcast_arrays = broadcast_arrays


def _patch_pytorch_round_numpy_contract() -> None:
    """Make raw and active public PyTorch round accept NumPy-style inputs."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    if getattr(pytorch_backend.round, "_pyrecest_numpy_contract", False):
        if active_pytorch_backend:
            backend.round = pytorch_backend.round
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
    pytorch_backend.round = round
    if active_pytorch_backend:
        backend.round = round


def _patch_raw_pytorch_conj_numpy_contract() -> None:
    """Make raw and active public PyTorch conj accept array-like inputs."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_conj = getattr(pytorch_backend, "conj", None)
    if original_conj is None:
        return
    if getattr(original_conj, "_pyrecest_numpy_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.conj = original_conj
        return

    def conj(a):
        return _torch.conj(pytorch_backend.array(a))

    conj.__name__ = getattr(original_conj, "__name__", "conj")
    conj.__doc__ = getattr(original_conj, "__doc__", None)
    conj._pyrecest_numpy_contract = True
    pytorch_backend.conj = conj
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.conj = conj


def _patch_pytorch_special_numpy_contract() -> None:
    """Make PyTorch special functions accept NumPy-style array-like inputs."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

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

    def _special_input(a):
        values = pytorch_backend.array(a)
        if not pytorch_backend.is_floating(values) and not pytorch_backend.is_complex(
            values
        ):
            values = pytorch_backend.cast(
                values, dtype=pytorch_backend.get_default_dtype()
            )
        return values

    def _gamma_from_lgamma(values):
        result = _torch.exp(_torch.special.gammaln(values))
        if pytorch_backend.is_complex(values):
            return result

        sign = _torch.ones_like(result)
        negative = values < 0
        negative_zero = (values == 0) & _torch.signbit(values)
        nonpositive_integer_pole = negative & (values == _torch.floor(values))
        reflected_sign = _torch.sign(_torch.sin(_torch.pi * values)).to(
            dtype=result.dtype
        )
        sign = _torch.where(negative, reflected_sign, sign)
        sign = _torch.where(negative_zero, -_torch.ones_like(sign), sign)
        result = result * sign
        return _torch.where(
            nonpositive_integer_pole,
            _torch.full_like(result, float("nan")),
            result,
        )

    def erf(a, out=None):
        result = _torch.erf(_special_input(a))
        return _return_or_store_out(result, out)

    def gammaln(a, out=None):
        result = _torch.special.gammaln(_special_input(a))
        return _return_or_store_out(result, out)

    def gamma(a, out=None):
        result = _gamma_from_lgamma(_special_input(a))
        return _return_or_store_out(result, out)

    def polygamma(n, a, out=None):
        result = _torch.polygamma(n, _special_input(a))
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
        setattr(pytorch_backend, name, helper)
        if active_pytorch_backend:
            setattr(backend, name, helper)

    pytorch_backend._gammaln = gammaln  # pylint: disable=protected-access


def _patch_pytorch_stack_helpers_numpy_contract() -> None:
    """Make PyTorch stack helpers accept NumPy-style array-like inputs."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import numpy as _np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    helper_names = ("stack", "hstack", "vstack", "column_stack", "dstack")
    if all(
        getattr(
            getattr(pytorch_backend, helper_name), "_pyrecest_numpy_contract", False
        )
        for helper_name in helper_names
    ):
        if active_pytorch_backend:
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(pytorch_backend, helper_name))
        return

    def _tensor_sequence(tup):
        return [pytorch_backend.array(item) for item in tup]

    def stack(arrays, axis=0, out=None, *, dim=None):
        if dim is not None:
            if axis != 0 and axis != dim:
                raise TypeError("stack() got both 'axis' and 'dim'")
            axis = dim
        tensors = _tensor_sequence(arrays)
        if not tensors:
            raise ValueError("need at least one array to stack")
        tensors = pytorch_backend.convert_to_wider_dtype(tensors)
        return _torch.stack(tensors, dim=_operator_index(axis), out=out)

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
        "stack": stack,
        "hstack": hstack,
        "vstack": vstack,
        "column_stack": column_stack,
        "dstack": dstack,
    }.items():
        helper.__name__ = helper_name
        helper.__doc__ = getattr(_np, helper_name).__doc__
        helper._pyrecest_numpy_contract = True
        setattr(pytorch_backend, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


def _normalize_pytorch_reduction_axis(axis):
    """Return Python int axes for scalar integer array/tensor axes."""

    if axis is None:
        return None
    try:
        return _operator_index(axis)
    except TypeError:
        return axis


def _wrap_pytorch_reduction_axis_helper(original_reduction):
    """Normalize scalar array axes before PyTorch reduction helpers see them."""

    if getattr(original_reduction, "_pyrecest_scalar_array_axis_contract", False):
        return original_reduction

    def reduction(a, axis=None, *args, **kwargs):
        if "dim" in kwargs:
            kwargs = dict(kwargs)
            kwargs["dim"] = _normalize_pytorch_reduction_axis(kwargs["dim"])
        axis = _normalize_pytorch_reduction_axis(axis)
        return original_reduction(a, axis, *args, **kwargs)

    reduction.__name__ = getattr(original_reduction, "__name__", "reduction")
    reduction.__doc__ = getattr(original_reduction, "__doc__", None)
    reduction._pyrecest_scalar_array_axis_contract = True
    return reduction


def _wrap_pytorch_quantile_axis_helper(original_quantile):
    """Normalize scalar array axes before PyTorch quantile helpers see them."""

    if getattr(original_quantile, "_pyrecest_scalar_array_axis_contract", False):
        return original_quantile

    def quantile(a, q, axis=None, *args, **kwargs):
        if "dim" in kwargs:
            kwargs = dict(kwargs)
            kwargs["dim"] = _normalize_pytorch_reduction_axis(kwargs["dim"])
        axis = _normalize_pytorch_reduction_axis(axis)
        return original_quantile(a, q, axis, *args, **kwargs)

    quantile.__name__ = getattr(original_quantile, "__name__", "quantile")
    quantile.__doc__ = getattr(original_quantile, "__doc__", None)
    quantile._pyrecest_scalar_array_axis_contract = True
    return quantile


def _patch_pytorch_reduction_axis_numpy_contract() -> None:
    """Make PyTorch reductions accept NumPy-style scalar array axes."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    for helper_name in (
        "all",
        "amax",
        "amin",
        "any",
        "argmax",
        "argmin",
        "count_nonzero",
        "max",
        "mean",
        "min",
        "prod",
        "std",
        "sum",
    ):
        helper = _wrap_pytorch_reduction_axis_helper(
            getattr(pytorch_backend, helper_name)
        )
        setattr(pytorch_backend, helper_name, helper)
        if active_pytorch_backend:
            setattr(
                backend,
                helper_name,
                _wrap_pytorch_reduction_axis_helper(getattr(backend, helper_name)),
            )

    quantile = _wrap_pytorch_quantile_axis_helper(pytorch_backend.quantile)
    pytorch_backend.quantile = quantile
    if active_pytorch_backend:
        backend.quantile = _wrap_pytorch_quantile_axis_helper(backend.quantile)


def _patch_raw_pytorch_comparison_numpy_contract() -> None:
    """Make raw PyTorch comparison helpers accept NumPy-style array-like inputs."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    def _coerce_binary_args(x, y):
        device = _preferred_pytorch_device(_torch, x, y)
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
        existing = getattr(pytorch_backend, helper_name, None)
        if getattr(existing, "_pyrecest_numpy_contract", False):
            if active_pytorch_backend:
                setattr(backend, helper_name, existing)
            return existing

        def comparison(x, y, **kwargs):
            x, y = _coerce_binary_args(x, y)
            return torch_func(x, y, **kwargs)

        comparison.__name__ = getattr(torch_func, "__name__", helper_name)
        comparison.__doc__ = getattr(torch_func, "__doc__", None)
        comparison._pyrecest_numpy_contract = True
        return comparison

    for helper_name, torch_func in (
        ("equal", _torch.eq),
        ("greater", _torch.greater),
        ("less", _torch.less),
        ("less_equal", _torch.le),
        ("logical_or", _torch.logical_or),
        ("logical_and", _torch.logical_and),
    ):
        helper = _wrap_comparison(helper_name, torch_func)
        setattr(pytorch_backend, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


def _patch_raw_pytorch_isclose_equal_nan_contract() -> None:
    """Make PyTorch isclose accept NumPy's ``equal_nan`` keyword."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    def isclose(
        x,
        y,
        rtol=pytorch_backend.rtol,
        atol=pytorch_backend.atol,
        equal_nan=False,
    ):
        device = _preferred_pytorch_device(_torch, x, y)
        if not _torch.is_tensor(x):
            x = _torch.as_tensor(x, device=device)
        elif device is not None and x.device != device:
            x = x.to(device=device)
        if not _torch.is_tensor(y):
            y = _torch.as_tensor(y, device=device)
        elif device is not None and y.device != device:
            y = y.to(device=device)
        x, y = pytorch_backend.convert_to_wider_dtype([x, y])
        return _torch.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

    isclose.__name__ = getattr(pytorch_backend.isclose, "__name__", "isclose")
    isclose.__doc__ = getattr(pytorch_backend.isclose, "__doc__", None)
    isclose._pyrecest_equal_nan_contract = True
    pytorch_backend.isclose = isclose
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.isclose = isclose


def _normalize_get_slice_indices(indices):
    """Return grouped get_slice indices in a backend-compatible form."""
    if isinstance(indices, tuple):
        return indices
    if isinstance(indices, (str, bytes)):
        return indices

    ndim = getattr(indices, "ndim", None)
    if ndim is not None:
        return tuple(indices) if ndim > 1 else indices

    if isinstance(indices, list):
        if not indices:
            return indices
        first_index = indices[0]
        if isinstance(first_index, (str, bytes)):
            return indices
        if hasattr(first_index, "__len__"):
            return tuple(indices)

    return indices


def _patch_get_slice_arraylike_contract() -> None:
    """Make public get_slice accept array-like inputs and grouped list indices."""
    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    original_get_slice = getattr(backend, "get_slice", None)
    array_func = getattr(backend, "array", None) or getattr(backend, "asarray", None)
    if original_get_slice is None or array_func is None:
        return
    if getattr(original_get_slice, "_pyrecest_get_slice_contract", False):
        return

    is_array_func = getattr(backend, "is_array", None)

    def get_slice(x, indices):
        if is_array_func is None or not is_array_func(x):
            x = array_func(x)
        return original_get_slice(x, _normalize_get_slice_indices(indices))

    get_slice.__name__ = getattr(original_get_slice, "__name__", "get_slice")
    get_slice.__doc__ = getattr(original_get_slice, "__doc__", None)
    get_slice._pyrecest_get_slice_contract = True
    backend.get_slice = get_slice


_patch_pytorch_assignment_scalar_tensor_indices()
_patch_pytorch_diag_numpy_contract()
_patch_pytorch_broadcast_arrays_numpy_contract()
_patch_pytorch_round_numpy_contract()
_patch_raw_pytorch_conj_numpy_contract()
_patch_pytorch_special_numpy_contract()
_patch_pytorch_stack_helpers_numpy_contract()
_patch_pytorch_reduction_axis_numpy_contract()
_patch_raw_pytorch_comparison_numpy_contract()
_patch_raw_pytorch_isclose_equal_nan_contract()
_patch_get_slice_arraylike_contract()


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
