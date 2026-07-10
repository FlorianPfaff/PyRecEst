"""Compatibility helpers for empty random uniform draws."""

from __future__ import annotations

import os


def _patch_pytorch_copy_export_contract() -> None:
    """Keep the PyTorch package-level ``copy`` export synchronized."""

    try:
        from pyrecest.backend_support._pytorch_copy_export_contract import (  # pylint: disable=import-outside-toplevel
            patch_pytorch_copy_export_contract,
        )
    except ModuleNotFoundError:  # pragma: no cover - backend support may be unavailable
        return
    patch_pytorch_copy_export_contract()


def _shape_has_no_samples(shape) -> bool:
    """Return whether a sample shape requests zero samples."""

    return any(int(dimension) == 0 for dimension in tuple(shape))


def _sample_shape_from_numpy_size(size, low_array, high_array, numpy_module):
    normalized_size = size
    if normalized_size is None:
        return tuple(numpy_module.broadcast_shapes(low_array.shape, high_array.shape))
    if isinstance(normalized_size, int):
        sample_shape = (normalized_size,)
    else:
        sample_shape = tuple(normalized_size)
    broadcast_shape = tuple(
        numpy_module.broadcast_shapes(sample_shape, low_array.shape, high_array.shape)
    )
    if broadcast_shape != sample_shape:
        raise ValueError("size, low, and high could not be broadcast together")
    return sample_shape


def _patch_shared_numpy_uniform_empty_bounds_contract() -> None:
    """Make shared NumPy-style ``uniform`` allow empty descending intervals."""

    try:
        import pyrecest._backend._shared_numpy.random as raw_random  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - shared backend should exist
        return

    try:
        import pyrecest._backend.numpy.random as numpy_random  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - NumPy backend may be unavailable
        numpy_random = None
    try:
        import pyrecest._backend.autograd.random as autograd_random  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - Autograd backend is optional
        autograd_random = None

    original_uniform = getattr(raw_random, "uniform", None)
    if original_uniform is None:
        return
    if getattr(original_uniform, "_pyrecest_empty_bounds_contract", False):
        if getattr(backend, "__backend_name__", None) in {"autograd", "numpy"}:
            backend.random.uniform = original_uniform
        return

    def _empty_result(low, high, size, args, kwargs):
        if args:
            return None
        dtype = kwargs.get("dtype", None)
        numpy_module = raw_random._np  # pylint: disable=protected-access
        low_array = (
            raw_random._validate_uniform_bound(  # pylint: disable=protected-access
                low,
                "low",
            )
        )
        high_array = (
            raw_random._validate_uniform_bound(  # pylint: disable=protected-access
                high,
                "high",
            )
        )
        sample_shape = _sample_shape_from_numpy_size(
            raw_random._normalize_size(size),  # pylint: disable=protected-access
            low_array,
            high_array,
            numpy_module,
        )
        if not _shape_has_no_samples(sample_shape):
            return None
        empty_samples = numpy_module.empty(sample_shape, dtype=dtype)
        return (high_array - low_array) * empty_samples + low_array

    def uniform(low=0.0, high=1.0, size=None, *args, **kwargs):
        try:
            return original_uniform(low, high, size, *args, **kwargs)
        except ValueError as exc:
            if "Upper bound must be greater than or equal to lower bound" not in str(
                exc
            ):
                raise
            result = _empty_result(low, high, size, args, kwargs)
            if result is None:
                raise
            return result

    uniform.__name__ = getattr(original_uniform, "__name__", "uniform")
    uniform.__doc__ = getattr(original_uniform, "__doc__", None)
    uniform._pyrecest_empty_bounds_contract = True
    raw_random.uniform = uniform
    if numpy_random is not None:
        numpy_random.uniform = uniform
    if autograd_random is not None:
        autograd_random.uniform = uniform
    if getattr(backend, "__backend_name__", None) in {"autograd", "numpy"}:
        backend.random.uniform = uniform


def _patch_pytorch_uniform_empty_bounds_contract() -> None:
    """Make PyTorch ``uniform`` allow empty descending intervals."""

    try:
        import pyrecest._backend.pytorch.random as raw_random  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_uniform = getattr(raw_random, "uniform", None)
    if original_uniform is None:
        return
    if getattr(original_uniform, "_pyrecest_empty_bounds_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.random.uniform = original_uniform
        return

    def _empty_result(low, high, size, dtype):
        dtype = raw_random._normalize_random_dtype(
            dtype, default=None
        )  # pylint: disable=protected-access
        device = None
        if torch.is_tensor(low):
            device = low.device
        elif torch.is_tensor(high):
            device = high.device
        low_tensor = (
            raw_random._validate_uniform_bound(  # pylint: disable=protected-access
                low,
                "low",
                dtype=dtype,
                device=device,
            )
        )
        high_tensor = (
            raw_random._validate_uniform_bound(  # pylint: disable=protected-access
                high,
                "high",
                dtype=dtype,
                device=device,
            )
        )
        sample_shape = raw_random._uniform_size(  # pylint: disable=protected-access
            size,
            low_tensor,
            high_tensor,
        )
        if not _shape_has_no_samples(sample_shape):
            return None
        empty_samples = torch.empty(sample_shape, dtype=dtype, device=device)
        return (high_tensor - low_tensor) * empty_samples + low_tensor

    def uniform(low=0.0, high=1.0, size=None, dtype=None):
        try:
            return original_uniform(low, high, size=size, dtype=dtype)
        except ValueError as exc:
            if "Upper bound must be greater than or equal to lower bound" not in str(
                exc
            ):
                raise
            result = _empty_result(low, high, size, dtype)
            if result is None:
                raise
            return result

    uniform.__name__ = getattr(original_uniform, "__name__", "uniform")
    uniform.__doc__ = getattr(original_uniform, "__doc__", None)
    uniform._pyrecest_empty_bounds_contract = True
    raw_random.uniform = uniform
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.random.uniform = uniform


def _patch_jax_uniform_empty_bounds_contract() -> None:
    """Make JAX ``uniform`` allow empty descending intervals."""

    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax.random as raw_random  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend may be unavailable
        return

    original_uniform = getattr(raw_random, "uniform", None)
    if original_uniform is None:
        return
    if getattr(original_uniform, "_pyrecest_empty_bounds_contract", False):
        if getattr(backend, "__backend_name__", None) == "jax":
            backend.random.uniform = original_uniform
        return

    def _empty_result(low, high, size, args, kwargs):
        if args:
            return None
        low_value, high_value = (
            raw_random._validate_uniform_bounds(  # pylint: disable=protected-access
                low,
                high,
            )
        )
        sample_shape = (
            raw_random._bounded_sampler_shape(  # pylint: disable=protected-access
                size,
                low_value,
                high_value,
            )
        )
        if not _shape_has_no_samples(sample_shape):
            return None
        state, has_state, remaining_kwargs = raw_random._get_state(
            **kwargs
        )  # pylint: disable=protected-access
        state, _ = raw_random.jax.random.split(state)
        dtype = remaining_kwargs.get("dtype", None)
        result = jnp.empty(sample_shape, dtype=dtype)
        return raw_random.set_state_return(has_state, state, result)

    def uniform(low=0.0, high=1.0, size=None, *args, **kwargs):
        try:
            return original_uniform(low, high, size, *args, **kwargs)
        except ValueError as exc:
            if "Upper bound must be greater than or equal to lower bound" not in str(
                exc
            ):
                raise
            result = _empty_result(low, high, size, args, kwargs)
            if result is None:
                raise
            return result

    uniform.__name__ = getattr(original_uniform, "__name__", "uniform")
    uniform.__doc__ = getattr(original_uniform, "__doc__", None)
    uniform._pyrecest_empty_bounds_contract = True
    raw_random.uniform = uniform
    if getattr(backend, "__backend_name__", None) == "jax":
        backend.random.uniform = uniform


def patch_random_uniform_empty_bounds_contract() -> None:
    """Patch random uniform empty-bound compatibility for optional backends."""

    _patch_pytorch_copy_export_contract()
    backend_name = os.environ.get("PYRECEST_BACKEND", "numpy")
    if backend_name in {"autograd", "numpy"}:
        _patch_shared_numpy_uniform_empty_bounds_contract()
    _patch_pytorch_uniform_empty_bounds_contract()
    _patch_jax_uniform_empty_bounds_contract()
