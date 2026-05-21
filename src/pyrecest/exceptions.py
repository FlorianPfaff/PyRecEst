"""Shared exception classes for PyRecEst.

These classes are intentionally small and dependency-free.  They provide a
stable vocabulary for user-facing failures without forcing every internal
validation helper to change at once.  Existing ``ValueError`` and
``NotImplementedError`` call sites can be migrated incrementally.
"""

from __future__ import annotations

from collections.abc import Iterable


class PyRecEstError(Exception):
    """Base class for PyRecEst-specific exceptions."""


class BackendNotSupportedError(NotImplementedError, PyRecEstError):
    """Raised when an API is unavailable for the active numerical backend.

    The class subclasses :class:`NotImplementedError` so existing tests and
    callers that check for unavailable operations continue to work while the
    message becomes more structured and actionable.
    """

    def __init__(
        self,
        api: str,
        backend: str,
        *,
        supported_backends: Iterable[str] | None = None,
        reason: str | None = None,
    ) -> None:
        self.api = api
        self.backend = backend
        self.supported_backends = tuple(supported_backends or ())
        self.reason = reason

        message = f"{api} is not supported for backend '{backend}'"
        if self.supported_backends:
            supported = ", ".join(self.supported_backends)
            message += f"; supported backends: {supported}"
        if reason:
            message += f"; reason: {reason}"
        super().__init__(message)


class ShapeError(ValueError, PyRecEstError):
    """Raised when an array, vector, matrix, or measurement set has bad shape."""

    def __init__(
        self,
        name: str,
        actual_shape: object,
        *,
        expected: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.name = name
        self.actual_shape = actual_shape
        self.expected = expected
        self.reason = reason

        message = f"{name} has invalid shape {actual_shape!r}"
        if expected:
            message += f"; expected {expected}"
        if reason:
            message += f"; reason: {reason}"
        super().__init__(message)


class DimensionMismatchError(ShapeError):
    """Raised when two or more objects have inconsistent dimensions."""

    def __init__(
        self, left_name: str, left_dim: int, right_name: str, right_dim: int
    ) -> None:
        self.left_name = left_name
        self.left_dim = left_dim
        self.right_name = right_name
        self.right_dim = right_dim
        super().__init__(
            f"{left_name}/{right_name}",
            (left_dim, right_dim),
            expected="matching dimensions",
            reason=f"{left_name} has dimension {left_dim}, but {right_name} has dimension {right_dim}",
        )


class NumericalStabilityError(PyRecEstError):
    """Raised when a numerically unstable operation cannot be completed safely."""

    def __init__(self, operation: str, *, reason: str | None = None) -> None:
        self.operation = operation
        self.reason = reason
        message = f"Numerical stability failure in {operation}"
        if reason:
            message += f": {reason}"
        super().__init__(message)
