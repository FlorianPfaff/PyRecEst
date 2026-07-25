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

    return (
        isinstance(indices, tuple)
        and bool(indices)
        and _is_array_like_index(indices[0], np, jnp)
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


def _rectangular_triangular_to_vec(helper_name, original_helper, index_helper, jnp):
    """Return a triangular-vector helper that uses both trailing matrix axes."""

    if getattr(original_helper, "_pyrecest_rectangular_contract", False):
        return original_helper

    def triangular_to_vec(x, k=0):
        x = jnp.asarray(x)
        if x.ndim < 2:
            raise ValueError(
                "triangular vector helpers require at least two matrix dimensions"
            )
        rows, cols = index_helper(x.shape[-2], k=k, m=x.shape[-1])
        return x[..., rows, cols]

    triangular_to_vec.__name__ = getattr(original_helper, "__name__", helper_name)
    triangular_to_vec.__doc__ = getattr(original_helper, "__doc__", None)
    triangular_to_vec._pyrecest_arraylike_contract = True
    triangular_to_vec._pyrecest_rectangular_contract = True
    return triangular_to_vec


def _patch_jax_triangular_vector_helpers_rectangular_contract(
    jax_backend, backend, jnp
) -> None:
    """Make JAX triangular-vector helpers work for rectangular matrices."""

    active_jax_backend = getattr(backend, "__backend_name__", None) == "jax"
    for helper_name, index_helper_name in (
        ("tril_to_vec", "tril_indices"),
        ("triu_to_vec", "triu_indices"),
    ):
        original_helper = getattr(jax_backend, helper_name, None)
        index_helper = getattr(jnp, index_helper_name, None)
        if original_helper is None or index_helper is None:
            continue

        helper = _rectangular_triangular_to_vec(
            helper_name,
            original_helper,
            index_helper,
            jnp,
        )
        setattr(jax_backend, helper_name, helper)
        if active_jax_backend:
            setattr(backend, helper_name, helper)


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

    _patch_jax_triangular_vector_helpers_rectangular_contract(jax_backend, backend, jnp)
