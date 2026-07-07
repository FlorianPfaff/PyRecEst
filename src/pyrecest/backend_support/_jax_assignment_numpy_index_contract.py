"""JAX assignment compatibility helpers."""

from __future__ import annotations


def _normalize_indices(indices, np, jnp):
    """Return JAX-friendly index arrays for NumPy ndarray inputs."""

    if isinstance(indices, np.ndarray):
        if indices.ndim > 0 and indices.size == 0:
            return indices
        return jnp.asarray(indices)
    return indices


def _is_array_like_index(index, np, jnp):
    """Return whether one index entry is a non-scalar array-like index."""

    if isinstance(index, np.ndarray):
        return index.ndim > 0
    if isinstance(index, jnp.ndarray):
        return index.ndim > 0
    return isinstance(index, (list, tuple))


def _is_per_axis_tuple_index(indices, np, jnp):
    """Return whether ``indices`` is a NumPy-style tuple of per-axis arrays."""

    return isinstance(indices, tuple) and bool(indices) and _is_array_like_index(
        indices[0], np, jnp
    )


def _as_per_axis_tuple(indices, jnp):
    """Coerce a tuple of per-axis index arrays to JAX arrays."""

    return tuple(jnp.asarray(index_axis) for index_axis in indices)


def _wrap_helper(helper, np, jnp):
    """Normalize NumPy ndarray indices before delegating to a JAX helper."""

    if getattr(helper, "_pyrecest_numpy_index_contract", False):
        return helper

    helper_name = getattr(helper, "__name__", "assignment")
    is_sum_helper = helper_name == "assignment_by_sum"

    def wrapped(x, values, indices, axis=0):
        if _is_per_axis_tuple_index(indices, np, jnp):
            target = jnp.asarray(x)
            normalized_indices = _as_per_axis_tuple(indices, jnp)
            updater = target.at[normalized_indices]
            if is_sum_helper:
                return updater.add(values)
            return updater.set(values)
        return helper(x, values, _normalize_indices(indices, np, jnp), axis=axis)

    wrapped.__name__ = helper_name
    wrapped.__doc__ = getattr(helper, "__doc__", None)
    wrapped._pyrecest_numpy_index_contract = True
    return wrapped


def patch_jax_assignment_numpy_index_contract() -> None:
    """Make JAX assignment helpers accept NumPy ndarray index sequences."""

    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax as jax_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend may be unavailable
        return

    jax_backend.assignment = _wrap_helper(jax_backend.assignment, np, jnp)
    jax_backend.assignment_by_sum = _wrap_helper(jax_backend.assignment_by_sum, np, jnp)
    if getattr(backend, "__backend_name__", None) == "jax":
        backend.assignment = jax_backend.assignment
        backend.assignment_by_sum = jax_backend.assignment_by_sum
