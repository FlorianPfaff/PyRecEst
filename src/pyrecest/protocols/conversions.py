"""Public protocols for distribution representation conversion."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable

ConversionTarget: TypeAlias = type[Any] | str
"""Target accepted by PyRecEst conversion helpers.

Concrete distribution classes and registered string aliases are both accepted by
``convert_distribution`` and distribution convenience wrappers.
"""

ConversionParameters: TypeAlias = dict[str, Any]
"""Keyword parameters passed to a conversion route."""


@runtime_checkable
class SupportsFromDistribution(Protocol):
    """Target representation that can build itself from a source distribution.

    Classes satisfying this protocol can be passed as concrete targets to
    ``pyrecest.distributions.conversion.convert_distribution`` when no explicit
    registered conversion route is required.
    """

    @classmethod
    def from_distribution(
        cls, distribution: Any, /, *args: Any, **kwargs: Any
    ) -> Any:
        """Create a target representation from ``distribution``."""
        ...


@runtime_checkable
class SupportsConvertTo(Protocol):
    """Object exposing the exact-or-approximate conversion convenience method."""

    def convert_to(
        self,
        target_type: ConversionTarget,
        /,
        *,
        return_info: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Convert or approximate this object as ``target_type``."""
        ...


@runtime_checkable
class SupportsApproximateAs(Protocol):
    """Object exposing an explicitly approximation-oriented conversion method."""

    def approximate_as(
        self,
        target_type: ConversionTarget,
        /,
        *,
        return_info: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Approximate this object as ``target_type``."""
        ...


@runtime_checkable
class SupportsDistributionConversion(
    SupportsConvertTo, SupportsApproximateAs, Protocol
):
    """Object exposing both PyRecEst conversion convenience wrappers."""


@runtime_checkable
class DistributionConverter(Protocol):
    """Callable registered as a concrete source-to-target conversion route."""

    def __call__(self, distribution: Any, /, **kwargs: Any) -> Any:
        """Convert ``distribution`` using keyword conversion parameters."""
        ...


@runtime_checkable
class ConversionAliasResolver(Protocol):
    """Callable resolving a conversion alias to a concrete target class."""

    def __call__(self, distribution: Any, /) -> type[Any]:
        """Return the concrete target class for ``distribution``."""
        ...


__all__ = [
    "ConversionAliasResolver",
    "ConversionParameters",
    "ConversionTarget",
    "DistributionConverter",
    "SupportsApproximateAs",
    "SupportsConvertTo",
    "SupportsDistributionConversion",
    "SupportsFromDistribution",
]
