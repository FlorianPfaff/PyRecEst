"""PyTorch creation-helper scalar and shape validation hook."""

from __future__ import annotations

from operator import index as _operator_index

_ARANGE_SENTINEL = object()


def _pytorch_has_temporal_dtype(value, numpy_module) -> bool:
    """Return whether ``value`` has a NumPy temporal dtype."""
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    try:
        return bool(
            numpy_module.issubdtype(dtype, numpy_module.datetime64)
            or numpy_module.issubdtype(dtype, numpy_module.timedelta64)
        )
    except TypeError:
        dtype_name = str(dtype).lower()
        return "datetime64" in dtype_name or "timedelta64" in dtype_name


def _pytorch_creation_dimension(dimension, numpy_module) -> int:
    """Return one NumPy-style shape dimension as a non-boolean integer."""
    if isinstance(dimension, (bool, numpy_module.bool_)):
        raise TypeError("shape dimensions must be integers")
    try:
        return _operator_index(dimension)
    except TypeError as exc:
        raise TypeError("shape dimensions must be integers") from exc


def _pytorch_creation_shape(shape, numpy_module, torch_module) -> tuple[int, ...]:
    """Normalize NumPy-style creation shapes while rejecting boolean dimensions."""
    if torch_module.is_tensor(shape):
        shape = shape.detach().cpu().numpy()
    if isinstance(shape, (list, tuple)):
        normalized_shape = tuple(
            _pytorch_creation_dimension(one_dimension, numpy_module)
            for one_dimension in shape
        )
        if any(one_dimension < 0 for one_dimension in normalized_shape):
            raise ValueError("negative dimensions are not allowed")
        return normalized_shape
    shape_array = numpy_module.asarray(shape)
    if _pytorch_has_temporal_dtype(shape_array, numpy_module):
        raise TypeError("shape dimensions must be integers")
    if shape_array.shape == ():
        normalized_shape = (
            _pytorch_creation_dimension(shape_array.item(), numpy_module),
        )
    else:
        normalized_shape = tuple(
            _pytorch_creation_dimension(one_dimension, numpy_module)
            for one_dimension in shape_array.tolist()
        )
    if any(one_dimension < 0 for one_dimension in normalized_shape):
        raise ValueError("negative dimensions are not allowed")
    return normalized_shape


def _pytorch_creation_scalar(value, numpy_module, torch_module, *, argument_name: str):
    """Normalize one NumPy-style scalar creation argument."""
    if torch_module.is_tensor(value):
        if value.ndim != 0:
            raise TypeError(f"{argument_name} must be a scalar")
        return value.item()
    value_array = numpy_module.asarray(value)
    if value_array.shape != ():
        raise TypeError(f"{argument_name} must be a scalar")
    if _pytorch_has_temporal_dtype(value_array, numpy_module):
        raise TypeError(f"{argument_name} must be numeric")
    return value_array.item()


def patch_pytorch_creation_shape_contract() -> None:
    """Patch raw/public PyTorch creation helpers for NumPy-style inputs."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
        from pyrecest._backend.pytorch._common import (  # pylint: disable=import-outside-toplevel
            _normalize_dtype,
        )
        from pyrecest.backend_support._pytorch_scatter_add_contract import (  # pylint: disable=import-outside-toplevel
            patch_pytorch_scatter_add_contract,
        )
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    patch_pytorch_scatter_add_contract()

    original_arange = getattr(raw_pytorch, "arange", None)
    if original_arange is not None:
        if getattr(original_arange, "_pyrecest_numpy_scalar_contract", False):
            if active_pytorch_backend:
                setattr(backend, "arange", original_arange)
        else:

            def arange(start, end=_ARANGE_SENTINEL, step=1, *, dtype=None, **kwargs):
                start = _pytorch_creation_scalar(
                    start,
                    np,
                    torch,
                    argument_name="arange start",
                )
                if end is _ARANGE_SENTINEL:
                    return torch.arange(
                        start,
                        dtype=_normalize_dtype(dtype),
                        **kwargs,
                    )
                end = _pytorch_creation_scalar(
                    end,
                    np,
                    torch,
                    argument_name="arange end",
                )
                step = _pytorch_creation_scalar(
                    step,
                    np,
                    torch,
                    argument_name="arange step",
                )
                return torch.arange(
                    start,
                    end,
                    step,
                    dtype=_normalize_dtype(dtype),
                    **kwargs,
                )

            arange.__name__ = getattr(original_arange, "__name__", "arange")
            arange.__doc__ = getattr(original_arange, "__doc__", None)
            arange._pyrecest_numpy_contract = True
            arange._pyrecest_numpy_scalar_contract = True
            setattr(raw_pytorch, "arange", arange)
            if active_pytorch_backend:
                setattr(backend, "arange", arange)

    def _wrap_creation_helper(helper_name, torch_helper, *, has_fill_value=False):
        original_helper = getattr(raw_pytorch, helper_name, None)
        if original_helper is None:
            return None
        if getattr(original_helper, "_pyrecest_boolean_shape_contract", False):
            if active_pytorch_backend:
                setattr(backend, helper_name, original_helper)
            return original_helper

        if has_fill_value:

            def creation_helper(shape, fill_value, dtype=None, *args, **kwargs):
                fill_value = _pytorch_creation_scalar(
                    fill_value,
                    np,
                    torch,
                    argument_name="full fill_value",
                )
                return torch_helper(
                    _pytorch_creation_shape(shape, np, torch),
                    fill_value,
                    *args,
                    dtype=_normalize_dtype(dtype),
                    **kwargs,
                )

        else:

            def creation_helper(shape, dtype=None, *args, **kwargs):
                return torch_helper(
                    _pytorch_creation_shape(shape, np, torch),
                    *args,
                    dtype=_normalize_dtype(dtype),
                    **kwargs,
                )

        creation_helper.__name__ = getattr(original_helper, "__name__", helper_name)
        creation_helper.__doc__ = getattr(original_helper, "__doc__", None)
        creation_helper._pyrecest_numpy_contract = True
        creation_helper._pyrecest_boolean_shape_contract = True
        return creation_helper

    def _wrap_like_creation_helper(helper_name, torch_helper, *, has_fill_value=False):
        original_helper = getattr(raw_pytorch, helper_name, None)
        if original_helper is None:
            return None
        if getattr(original_helper, "_pyrecest_like_creation_contract", False):
            if active_pytorch_backend:
                setattr(backend, helper_name, original_helper)
            return original_helper

        if has_fill_value:

            def like_creation_helper(a, fill_value, dtype=None, *args, **kwargs):
                return torch_helper(
                    raw_pytorch.array(a),
                    fill_value,
                    *args,
                    dtype=_normalize_dtype(dtype),
                    **kwargs,
                )

        else:

            def like_creation_helper(a, dtype=None, *args, **kwargs):
                return torch_helper(
                    raw_pytorch.array(a),
                    *args,
                    dtype=_normalize_dtype(dtype),
                    **kwargs,
                )

        like_creation_helper.__name__ = getattr(
            original_helper, "__name__", helper_name
        )
        like_creation_helper.__doc__ = getattr(original_helper, "__doc__", None)
        like_creation_helper._pyrecest_numpy_contract = True
        like_creation_helper._pyrecest_arraylike_contract = True
        like_creation_helper._pyrecest_like_creation_contract = True
        return like_creation_helper

    for helper_name, torch_helper, has_fill_value in (
        ("empty", torch.empty, False),
        ("zeros", torch.zeros, False),
        ("ones", torch.ones, False),
        ("full", torch.full, True),
    ):
        helper = _wrap_creation_helper(
            helper_name,
            torch_helper,
            has_fill_value=has_fill_value,
        )
        if helper is None:
            continue
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)

    for helper_name, torch_helper, has_fill_value in (
        ("empty_like", torch.empty_like, False),
        ("zeros_like", torch.zeros_like, False),
        ("ones_like", torch.ones_like, False),
        ("full_like", torch.full_like, True),
    ):
        helper = _wrap_like_creation_helper(
            helper_name,
            torch_helper,
            has_fill_value=has_fill_value,
        )
        if helper is None:
            continue
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)
