"""Small deprecation helper for public API transitions."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def deprecated(
    *,
    since: str,
    remove_in: str,
    replacement: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function or method with a standardized deprecation warning.

    Parameters
    ----------
    since:
        Version in which the API was deprecated.
    remove_in:
        Planned version in which the API may be removed.
    replacement:
        Optional replacement API name.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        message = (
            f"{func.__module__}.{func.__qualname__} is deprecated since "
            f"PyRecEst {since} and may be removed in PyRecEst {remove_in}."
        )
        if replacement:
            message += f" Use {replacement} instead."

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        wrapper.__deprecated_since__ = since  # type: ignore[attr-defined]
        wrapper.__deprecated_remove_in__ = remove_in  # type: ignore[attr-defined]
        wrapper.__deprecated_replacement__ = replacement  # type: ignore[attr-defined]
        return wrapper

    return decorator


__all__ = ["deprecated"]
