"""Validation helpers for reusable model objects."""

from __future__ import annotations

from typing import Any, TypeVar

CapabilityT = TypeVar("CapabilityT")


def validate_optional_dimension(name: str, value: int | None) -> int | None:
    """Validate an optional positive integer dimension."""
    if value is None:
        return None
    if not isinstance(value, int):
        raise TypeError("Dimension must be None or an integer.")
    if value < 1:
        raise ValueError("Dimension must be positive when specified.")
    return value


def require_capability(model: Any, capability: type[CapabilityT], capability_name: str | None = None) -> CapabilityT:
    """Return ``model`` when it implements ``capability``; otherwise raise."""
    if isinstance(model, capability):
        return model

    name = capability_name or getattr(capability, "__name__", str(capability))
    raise TypeError(f"Model object does not implement required capability {name}.")
