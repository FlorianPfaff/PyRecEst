"""Reproducibility helpers for backend-managed random state."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from contextlib import contextmanager
from numbers import Integral
from typing import Any

import numpy as np

_MAX_BACKEND_SEED = 2**32 - 1
_UNSUPPORTED_SEED_SCALAR_TYPES = (
    bool,
    str,
    bytes,
    bytearray,
    np.bool_,
    np.str_,
    np.bytes_,
)
_TEMPORAL_SEED_SCALAR_TYPES = (np.datetime64, np.timedelta64)
_TEMPORAL_SEED_DTYPE_KINDS = "Mm"


def _random_backend() -> Any:
    from pyrecest.backend import random

    return random


def _is_temporal_seed_scalar(seed: Any) -> bool:
    if isinstance(seed, _TEMPORAL_SEED_SCALAR_TYPES):
        return True
    try:
        seed_array = np.asarray(seed)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return False
    if seed_array.shape != ():
        return False
    if seed_array.dtype.kind in _TEMPORAL_SEED_DTYPE_KINDS:
        return True
    if seed_array.dtype != object:
        return False
    try:
        scalar = seed_array.item()
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return False
    return isinstance(scalar, _TEMPORAL_SEED_SCALAR_TYPES)


def _normalize_seed(seed: int | None) -> int | None:
    """Return a validated backend seed or ``None``."""

    if seed is None:
        return None

    message = "seed must be a non-negative integer or None"

    if _is_temporal_seed_scalar(seed):
        raise ValueError(message)

    if hasattr(seed, "shape") and tuple(seed.shape) != ():
        raise ValueError(message)

    try:
        scalar = seed.item() if hasattr(seed, "item") else seed
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(message) from exc

    if isinstance(scalar, _UNSUPPORTED_SEED_SCALAR_TYPES):
        raise ValueError(message)
    if isinstance(scalar, Integral):
        normalized_seed = int(scalar)
    else:
        try:
            scalar_float = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if not math.isfinite(scalar_float) or not scalar_float.is_integer():
            raise ValueError(message)
        normalized_seed = int(scalar_float)

    if normalized_seed < 0 or normalized_seed > _MAX_BACKEND_SEED:
        raise ValueError(message)
    return normalized_seed


def seed_all(seed: int | None) -> int | None:
    """Seed the active backend random module and return the normalized seed.

    PyRecEst uses backend-specific RNG implementations.  This helper gives
    scenario runners and tests one explicit entry point instead of relying on
    ad-hoc calls to ``pyrecest.backend.random.seed``.
    """
    normalized_seed = _normalize_seed(seed)
    if normalized_seed is None:
        return None
    _random_backend().seed(normalized_seed)
    return normalized_seed


def get_backend_random_state() -> Any:
    """Return the active backend random state when the backend exposes it."""
    random = _random_backend()
    if not hasattr(random, "get_state"):
        raise AttributeError(
            "The active backend random module does not expose get_state()."
        )
    return random.get_state()


def set_backend_random_state(state: Any) -> None:
    """Restore the active backend random state when the backend exposes it."""
    random = _random_backend()
    if not hasattr(random, "set_state"):
        raise AttributeError(
            "The active backend random module does not expose set_state()."
        )
    random.set_state(state)


@contextmanager
def preserve_backend_random_state() -> Iterator[None]:
    """Temporarily preserve backend and NumPy random state."""
    state = copy.deepcopy(get_backend_random_state())
    numpy_state = copy.deepcopy(np.random.get_state())
    try:
        yield
    finally:
        try:
            set_backend_random_state(state)
        finally:
            np.random.set_state(numpy_state)


@contextmanager
def temporary_seed(seed: int | None) -> Iterator[None]:
    """Run a block with ``seed`` and restore the previous backend RNG state."""
    with preserve_backend_random_state():
        seed_all(seed)
        yield


__all__ = [
    "get_backend_random_state",
    "preserve_backend_random_state",
    "seed_all",
    "set_backend_random_state",
    "temporary_seed",
]
