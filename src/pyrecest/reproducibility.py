"""Reproducibility helpers for backend-managed random state."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def _random_backend() -> Any:
    from pyrecest.backend import random

    return random


def seed_all(seed: int | None) -> int | None:
    """Seed the active backend random module and return the normalized seed.

    PyRecEst uses backend-specific RNG implementations.  This helper gives
    scenario runners and tests one explicit entry point instead of relying on
    ad-hoc calls to ``pyrecest.backend.random.seed``.
    """
    if seed is None:
        return None
    normalized_seed = int(seed)
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
    """Temporarily preserve and restore the active backend random state."""
    state = copy.deepcopy(get_backend_random_state())
    try:
        yield
    finally:
        set_backend_random_state(state)


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
