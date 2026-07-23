"""Small deprecation helper for public API transitions."""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def _validate_nonempty_version(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


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

    since = _validate_nonempty_version(since, "since")
    remove_in = _validate_nonempty_version(remove_in, "remove_in")
    if replacement is not None:
        replacement = _validate_nonempty_version(replacement, "replacement")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        message = (
            f"{func.__module__}.{func.__qualname__} is deprecated since "
            f"PyRecEst {since} and may be removed in PyRecEst {remove_in}."
        )
        if replacement:
            message += f" Use {replacement} instead."

        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def async_generator_wrapper(*args: P.args, **kwargs: P.kwargs):
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                async for item in func(*args, **kwargs):  # type: ignore[attr-defined]
                    yield item

            async_generator_wrapper.__deprecated_since__ = since  # type: ignore[attr-defined]
            async_generator_wrapper.__deprecated_remove_in__ = remove_in  # type: ignore[attr-defined]
            async_generator_wrapper.__deprecated_replacement__ = replacement  # type: ignore[attr-defined]
            return async_generator_wrapper  # type: ignore[return-value]

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs):
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                return await func(*args, **kwargs)

            async_wrapper.__deprecated_since__ = since  # type: ignore[attr-defined]
            async_wrapper.__deprecated_remove_in__ = remove_in  # type: ignore[attr-defined]
            async_wrapper.__deprecated_replacement__ = replacement  # type: ignore[attr-defined]
            return async_wrapper  # type: ignore[return-value]

        if inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def generator_wrapper(*args: P.args, **kwargs: P.kwargs):
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                return (yield from func(*args, **kwargs))  # type: ignore[misc]

            generator_wrapper.__deprecated_since__ = since  # type: ignore[attr-defined]
            generator_wrapper.__deprecated_remove_in__ = remove_in  # type: ignore[attr-defined]
            generator_wrapper.__deprecated_replacement__ = replacement  # type: ignore[attr-defined]
            return generator_wrapper  # type: ignore[return-value]

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
