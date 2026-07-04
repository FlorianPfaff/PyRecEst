"""Shared exception classes for PyRecEst.

These classes are intentionally small and dependency-free. They provide a
stable vocabulary for user-facing failures without forcing every internal
validation helper to change at once. Existing ``ValueError`` and
``NotImplementedError`` call sites can be migrated incrementally.
"""

from __future__ import annotations

from collections.abc import Iterable


def _decode_text_bytes(value: bytes | bytearray) -> str:
    try:
        return bytes(value).decode()
    except UnicodeDecodeError:
        return repr(bytes(value))


def _normalize_backend_name(backend: object) -> str:
    if isinstance(backend, (bytes, bytearray)):
        return _decode_text_bytes(backend)
    return str(backend)


def _normalize_supported_backends(
    supported_backends: Iterable[str] | str | None,
) -> tuple[str, ...]:
    if supported_backends is None:
        return ()
    if isinstance(supported_backends, str):
        return (supported_backends,)
    if isinstance(supported_backends, (bytes, bytearray)):
        return (_decode_text_bytes(supported_backends),)
    return tuple(_normalize_backend_name(backend) for backend in supported_backends)


class PyRecEstError(Exception):
    """Base class for PyRecEst-specific exceptions."""


class BackendSupportError(PyRecEstError):
    """Base class for backend-selection and backend-capability errors."""


class BackendNotSupportedError(BackendSupportError, NotImplementedError):
    """Raised when an API is unavailable for the active numerical backend."""

    def __init__(
        self,
        api: str,
        backend: str | None = None,
        *,
        supported_backends: Iterable[str] | str | None = None,
        reason: str | None = None,
    ) -> None:
        self.api = api
        self.backend = backend
        self.supported_backends = _normalize_supported_backends(supported_backends)
        self.reason = reason

        if backend is None:
            message = api
        else:
            message = f"{api} is unavailable for backend '{backend}'"
        if self.supported_backends:
            supported = ", ".join(self.supported_backends)
            message += f"; supported backends: {supported}"
        if reason:
            message += f"; reason: {reason}"
        super().__init__(message)


class OptionalDependencyError(PyRecEstError, ImportError):
    """Raised when an optional extra is required for a feature."""


class ValidationError(PyRecEstError, ValueError):
    """Base class for PyRecEst input validation errors."""


class ShapeError(ValidationError):
    """Raised when an array, vector, matrix, or measurement set has bad shape."""

    def __init__(
        self,
        name: str,
        actual_shape: object | None = None,
        *,
        expected: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.name = name
        self.actual_shape = actual_shape
        self.expected = expected
        self.reason = reason

        if actual_shape is None:
            message = name
        else:
            message = f"{name} has invalid shape {actual_shape!r}"
        if expected:
            message += f"; expected {expected}"
        if reason:
            message += f"; reason: {reason}"
        super().__init__(message)


class DimensionMismatchError(ShapeError):
    """Raised when two or more objects have inconsistent dimensions."""

    def __init__(
        self,
        left_name: str,
        left_dim: int | None = None,
        right_name: str | None = None,
        right_dim: int | None = None,
    ) -> None:
        self.left_name = left_name
        self.left_dim = left_dim
        self.right_name = right_name
        self.right_dim = right_dim

        if left_dim is None or right_name is None or right_dim is None:
            super().__init__(left_name)
            return

        super().__init__(
            f"{left_name}/{right_name}",
            (left_dim, right_dim),
            expected="matching dimensions",
            reason=(
                f"{left_name} has dimension {left_dim}, but {right_name} has dimension {right_dim}"
            ),
        )


class NumericalStabilityError(ValidationError):
    """Raised when a numerically unstable operation cannot be completed safely."""

    def __init__(self, operation: str, *, reason: str | None = None) -> None:
        self.operation = operation
        self.reason = reason
        message = (
            operation
            if reason is None
            else f"Numerical stability failure in {operation}: {reason}"
        )
        super().__init__(message)


__all__ = [
    "BackendNotSupportedError",
    "BackendSupportError",
    "DimensionMismatchError",
    "NumericalStabilityError",
    "OptionalDependencyError",
    "PyRecEstError",
    "ShapeError",
    "ValidationError",
]
