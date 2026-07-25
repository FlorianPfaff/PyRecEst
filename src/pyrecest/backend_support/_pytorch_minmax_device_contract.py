"""PyTorch binary-helper device compatibility hook."""

from __future__ import annotations

import importlib
from operator import index as _operator_index

import numpy as _np

_AXIS_TYPE_ERROR = "axis must be None, an integer, or a tuple of integers"


def _preferred_pytorch_device(torch_module, *values):
    """Return a non-CPU tensor device when mixed-device operands are present."""
    for value in values:
        if torch_module.is_tensor(value) and value.device.type == "meta":
            return value.device
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def _binary_operands(raw_pytorch, torch_module, left, right):
    """Return operands on a common dtype and an existing preferred device."""
    device = _preferred_pytorch_device(torch_module, left, right)
    left = raw_pytorch.array(left)
    right = raw_pytorch.array(right)
    dtype = torch_module.promote_types(left.dtype, right.dtype)
    if device is None:
        return left.to(dtype=dtype), right.to(dtype=dtype)
    return left.to(device=device, dtype=dtype), right.to(device=device, dtype=dtype)


def _minmax_operands(raw_pytorch, torch_module, left, right):
    """Return extrema operands using NumPy-compatible dtype promotion."""
    device = _preferred_pytorch_device(torch_module, left, right)
    left = raw_pytorch.array(left)
    right = raw_pytorch.array(right)
    try:
        dtype = torch_module.promote_types(left.dtype, right.dtype)
    except RuntimeError as promotion_error:
        try:
            left_numpy_dtype = (
                torch_module.empty((), dtype=left.dtype, device="cpu").numpy().dtype
            )
            right_numpy_dtype = (
                torch_module.empty((), dtype=right.dtype, device="cpu").numpy().dtype
            )
            promoted_numpy_dtype = _np.result_type(
                left_numpy_dtype,
                right_numpy_dtype,
            )
            dtype = torch_module.from_numpy(
                _np.empty((), dtype=promoted_numpy_dtype)
            ).dtype
        except (TypeError, RuntimeError):
            raise promotion_error
    if device is None:
        return left.to(dtype=dtype), right.to(dtype=dtype)
    return left.to(device=device, dtype=dtype), right.to(device=device, dtype=dtype)


def _copy_to_out(result, out):
    """Store ``result`` in ``out`` and return the output buffer."""
    copy_ = getattr(out, "copy_", None)
    if copy_ is not None:
        copy_(result)
        return out
    out[...] = result
    return out


def _raw_pytorch_module():
    """Return the raw PyTorch backend module, importing it when available."""
    try:
        return importlib.import_module("pyrecest._backend.pytorch")
    except ModuleNotFoundError:
        return None


def _pytorch_norm_axis_entry(axis, torch_module) -> int:
    """Return one non-boolean NumPy-style norm axis."""
    if isinstance(axis, (bool, _np.bool_)):
        raise TypeError(_AXIS_TYPE_ERROR)
    if torch_module.is_tensor(axis):
        if axis.dtype == torch_module.bool:
            raise TypeError(_AXIS_TYPE_ERROR)
        if axis.ndim != 0:
            raise TypeError(_AXIS_TYPE_ERROR)
        return _operator_index(axis.item())
    if isinstance(axis, _np.ndarray):
        if axis.dtype == _np.bool_:
            raise TypeError(_AXIS_TYPE_ERROR)
        if axis.shape != ():
            raise TypeError(_AXIS_TYPE_ERROR)
        return _operator_index(axis.item())
    return _operator_index(axis)


def _normalize_pytorch_norm_axis(axis, torch_module):
    """Normalize PyTorch ``linalg.norm`` axes without accepting booleans."""
    if axis is None:
        return None
    if isinstance(axis, (bool, _np.bool_)):
        raise TypeError(_AXIS_TYPE_ERROR)
    if torch_module.is_tensor(axis):
        if axis.dtype == torch_module.bool:
            raise TypeError(_AXIS_TYPE_ERROR)
        if axis.ndim == 0:
            return _operator_index(axis.item())
        if axis.ndim != 1:
            raise TypeError(_AXIS_TYPE_ERROR)
        axis = axis.detach().cpu().tolist()
    elif isinstance(axis, _np.ndarray):
        if axis.dtype == _np.bool_:
            raise TypeError(_AXIS_TYPE_ERROR)
        if axis.ndim == 0:
            return _operator_index(axis.item())
        if axis.ndim != 1:
            raise TypeError(_AXIS_TYPE_ERROR)
        axis = axis.tolist()
    elif isinstance(axis, (int, _np.integer)):
        return int(axis)

    if isinstance(axis, (list, tuple)):
        return tuple(
            _pytorch_norm_axis_entry(one_axis, torch_module) for one_axis in axis
        )
    return _pytorch_norm_axis_entry(axis, torch_module)


def _patch_pytorch_linalg_norm_axis_contract(
    raw_pytorch, backend, torch_module
) -> None:
    """Patch PyTorch linalg.norm axis normalization for boolean sequences."""
    raw_linalg = getattr(raw_pytorch, "linalg", None)
    if raw_linalg is None:
        return
    original_norm = getattr(raw_linalg, "norm", None)
    if original_norm is None:
        return
    if getattr(original_norm, "_pyrecest_bool_sequence_axis_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            try:
                backend_linalg = importlib.import_module("pyrecest.backend.linalg")
            except (
                ModuleNotFoundError
            ):  # pragma: no cover - backend import failed first
                return
            backend_linalg.norm = original_norm
        return

    def norm(x, ord=None, axis=None, keepdims=False):
        axis = _normalize_pytorch_norm_axis(axis, torch_module)
        return original_norm(x, ord=ord, axis=axis, keepdims=keepdims)

    norm.__name__ = getattr(original_norm, "__name__", "norm")
    norm.__doc__ = getattr(original_norm, "__doc__", None)
    norm._pyrecest_bool_sequence_axis_contract = True
    raw_linalg.norm = norm
    if getattr(backend, "__backend_name__", None) == "pytorch":
        try:
            backend_linalg = importlib.import_module("pyrecest.backend.linalg")
        except ModuleNotFoundError:  # pragma: no cover - backend import failed first
            return
        backend_linalg.norm = norm


def _patch_pytorch_rectangular_triangular_vector_contract(raw_pytorch, backend) -> None:
    """Patch triangular-vector helpers to respect rectangular matrix shapes."""
    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    def _make_triangular_to_vec(helper_name, index_helper, original_helper):
        def triangular_to_vec(x, k=0):
            x = raw_pytorch.array(x)
            rows, cols = index_helper(x.shape[-2], k=k, m=x.shape[-1])
            rows = rows.to(device=x.device)
            cols = cols.to(device=x.device)
            return x[..., rows, cols]

        triangular_to_vec.__name__ = getattr(original_helper, "__name__", helper_name)
        triangular_to_vec.__doc__ = getattr(original_helper, "__doc__", None)
        triangular_to_vec._pyrecest_numpy_contract = True
        triangular_to_vec._pyrecest_rectangular_triangular_contract = True
        return triangular_to_vec

    for helper_name, index_helper_name in (
        ("tril_to_vec", "tril_indices"),
        ("triu_to_vec", "triu_indices"),
    ):
        original_helper = getattr(raw_pytorch, helper_name, None)
        index_helper = getattr(raw_pytorch, index_helper_name, None)
        if original_helper is None or index_helper is None:
            continue
        if getattr(original_helper, "_pyrecest_rectangular_triangular_contract", False):
            if active_pytorch_backend:
                setattr(backend, helper_name, original_helper)
            continue

        helper = _make_triangular_to_vec(helper_name, index_helper, original_helper)
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


def _patch_binary_helpers(
    raw_pytorch,
    backend,
    torch_module,
    helpers,
    contract_attr,
    *,
    supports_out=False,
    operand_normalizer=_binary_operands,
):
    """Patch raw/public binary helpers to share dtype and preserve device."""
    if all(
        getattr(
            getattr(raw_pytorch, helper_name, None),
            contract_attr,
            False,
        )
        for helper_name in helpers
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for helper_name in helpers:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    for helper_name, torch_helper in helpers.items():
        original_helper = getattr(raw_pytorch, helper_name)

        if supports_out:

            def binary_helper(
                left,
                right,
                out=None,
                _torch_helper=torch_helper,
                _operand_normalizer=operand_normalizer,
            ):
                left, right = _operand_normalizer(
                    raw_pytorch,
                    torch_module,
                    left,
                    right,
                )
                result = _torch_helper(left, right)
                if out is None:
                    return result
                return _copy_to_out(result, out)

        else:

            def binary_helper(
                left,
                right,
                _torch_helper=torch_helper,
                _operand_normalizer=operand_normalizer,
            ):
                left, right = _operand_normalizer(
                    raw_pytorch,
                    torch_module,
                    left,
                    right,
                )
                return _torch_helper(left, right)

        binary_helper.__name__ = getattr(original_helper, "__name__", helper_name)
        binary_helper.__doc__ = getattr(original_helper, "__doc__", None)
        setattr(binary_helper, contract_attr, True)
        binary_helper._pyrecest_device_contract = True
        setattr(raw_pytorch, helper_name, binary_helper)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, helper_name, binary_helper)


def patch_pytorch_minmax_device_contract() -> None:
    """Patch raw/public PyTorch helpers to preserve backend contracts."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    raw_pytorch = _raw_pytorch_module()
    if raw_pytorch is None:  # pragma: no cover - backend import failed earlier
        return
    _patch_binary_helpers(
        raw_pytorch,
        backend,
        torch,
        {
            "maximum": torch.maximum,
            "minimum": torch.minimum,
        },
        "_pyrecest_minmax_device_contract",
        supports_out=True,
        operand_normalizer=_minmax_operands,
    )
    _patch_binary_helpers(
        raw_pytorch,
        backend,
        torch,
        {
            "equal": torch.eq,
            "less_equal": torch.le,
            "logical_and": torch.logical_and,
        },
        "_pyrecest_comparison_device_contract",
    )
    _patch_pytorch_linalg_norm_axis_contract(raw_pytorch, backend, torch)
    _patch_pytorch_rectangular_triangular_vector_contract(raw_pytorch, backend)
