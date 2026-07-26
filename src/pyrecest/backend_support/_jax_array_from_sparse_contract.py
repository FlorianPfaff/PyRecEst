"""Runtime patch for JAX sparse reconstruction indexing."""

from __future__ import annotations

import operator

import numpy as np

_BOOLEAN_TYPES = (bool, np.bool_)
_TEXT_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)


def _normalize_sparse_dimension(size) -> int:
    if isinstance(size, _BOOLEAN_TYPES):
        raise TypeError("target_shape dimensions must be integers")
    dtype = getattr(size, "dtype", None)
    if dtype is not None and str(dtype).lower() in {
        "bool",
        "bool_",
        "torch.bool",
    }:
        raise TypeError("target_shape dimensions must be integers")
    try:
        normalized = operator.index(size)
    except TypeError as exc:
        raise TypeError("target_shape dimensions must be integers") from exc
    if normalized < 0:
        raise ValueError("negative dimensions are not allowed")
    return normalized


def _normalize_sparse_target_shape(target_shape) -> tuple[int, ...]:
    if isinstance(target_shape, _TEXT_TYPES):
        raise TypeError("target_shape must be an integer or a sequence of integers")
    try:
        dimensions = tuple(target_shape)
    except TypeError:
        dimensions = (target_shape,)
    return tuple(_normalize_sparse_dimension(size) for size in dimensions)


def patch_jax_array_from_sparse_flat_index_contract() -> None:
    """Make JAX ``array_from_sparse`` accept flat indices for 1-D targets."""

    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax as raw_jax  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend may be unavailable
        return

    active_jax_backend = getattr(backend, "__backend_name__", None) == "jax"
    original_array_from_sparse = getattr(raw_jax, "array_from_sparse", None)
    if original_array_from_sparse is None:
        return
    if getattr(
        original_array_from_sparse,
        "_pyrecest_sparse_flat_index_contract",
        False,
    ):
        if active_jax_backend:
            backend.array_from_sparse = original_array_from_sparse
        return

    def _jax_sparse_indices(indices, target_shape):
        indices = jnp.array(indices)
        if indices.size == 0:
            return indices
        if indices.ndim == 1 and len(target_shape) == 1:
            return indices.reshape(-1, 1)
        if indices.ndim != 2 or indices.shape[1] != len(target_shape):
            raise ValueError("indices must have shape (n_entries, ndim)")
        return indices

    def array_from_sparse(indices, data, target_shape):
        target_shape = _normalize_sparse_target_shape(target_shape)
        indices = _jax_sparse_indices(indices, target_shape)
        data = jnp.array(data)
        out = jnp.zeros(target_shape, dtype=data.dtype)
        if indices.size == 0:
            if data.size != 0:
                raise ValueError("data must be empty when indices are empty")
            return out

        linear_indices = jnp.atleast_1d(jnp.ravel_multi_index(indices.T, target_shape))
        if linear_indices.size:
            linear_count = linear_indices.size
            _, reversed_first_positions = jnp.unique(
                linear_indices[::-1],
                return_index=True,
            )
            keep_positions = jnp.sort(linear_count - 1 - reversed_first_positions)
            linear_indices = linear_indices[keep_positions]
            if data.ndim > 0 and data.shape[0] == linear_count:
                data = data[keep_positions]
        out = out.reshape(-1).at[linear_indices].set(data).reshape(target_shape)
        return out

    array_from_sparse.__name__ = getattr(
        original_array_from_sparse,
        "__name__",
        "array_from_sparse",
    )
    array_from_sparse.__doc__ = getattr(original_array_from_sparse, "__doc__", None)
    array_from_sparse._pyrecest_sparse_flat_index_contract = True
    raw_jax.array_from_sparse = array_from_sparse
    if active_jax_backend:
        backend.array_from_sparse = array_from_sparse
