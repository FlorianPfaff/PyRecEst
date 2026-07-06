"""Strict population validation for backend random.choice wrappers."""

from __future__ import annotations

from functools import wraps


def _patch_pytorch_choice() -> None:
    try:
        import pyrecest._backend.pytorch.random as random_backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover
        return

    original = getattr(random_backend, "choice", None)
    if original is None or getattr(original, "_pyrecest_population_contract", False):
        return

    def population_size(a, axis):
        scalar_size = random_backend._integer_population_size(a)  # pylint: disable=protected-access
        if scalar_size is not None:
            return scalar_size
        values = a if torch.is_tensor(a) else torch.as_tensor(a)
        if values.ndim == 0:
            return None
        axis = random_backend._normalize_axis(axis, values.ndim)  # pylint: disable=protected-access
        return values.shape[axis]

    @wraps(original)
    def choice(a, size=None, replace=True, p=None, axis=0, shuffle=True):
        pop_size = population_size(a, axis)
        if pop_size is not None and pop_size <= 0:
            raise ValueError("a must be a positive integer or a non-empty array")
        return original(a, size=size, replace=replace, p=p, axis=axis, shuffle=shuffle)

    choice._pyrecest_population_contract = True
    random_backend.choice = choice


def _patch_jax_choice() -> None:
    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax.random as random_backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover
        return

    original = getattr(random_backend, "choice", None)
    if original is None or getattr(original, "_pyrecest_population_contract", False):
        return

    def population_size(a, kwargs):
        values = jnp.asarray(a)
        scalar_size = random_backend._integer_population_size(values)  # pylint: disable=protected-access
        if scalar_size is not None:
            return scalar_size
        if values.ndim == 0:
            return None
        axis = random_backend._normalize_choice_axis(kwargs.get("axis", 0), values.ndim)  # pylint: disable=protected-access
        return values.shape[axis]

    @wraps(original)
    def choice(a, size=None, replace=True, p=None, shuffle=True, *args, **kwargs):
        pop_size = population_size(a, kwargs)
        if pop_size is not None and pop_size <= 0:
            raise ValueError("a must be a positive integer or a non-empty array")
        return original(a, size, replace, p, shuffle, *args, **kwargs)

    choice._pyrecest_population_contract = True
    random_backend.choice = choice


def patch_random_choice_population_contract() -> None:
    _patch_pytorch_choice()
    _patch_jax_choice()
