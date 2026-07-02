"""JAX assignment compatibility helpers."""

from __future__ import annotations


def _normalize_indices(indices, np, jnp):
    """Return JAX-friendly index arrays for NumPy ndarray inputs."""

    if isinstance(indices, np.ndarray):
        if indices.ndim > 0 and indices.size == 0:
            return indices
        return jnp.asarray(indices)
    return indices


def _wrap_helper(helper, np, jnp):
    """Normalize NumPy ndarray indices before delegating to a JAX helper."""

    if getattr(helper, "_pyrecest_numpy_index_contract", False):
        return helper

    def wrapped(x, values, indices, axis=0):
        return helper(x, values, _normalize_indices(indices, np, jnp), axis=axis)

    wrapped.__name__ = getattr(helper, "__name__", "assignment")
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
