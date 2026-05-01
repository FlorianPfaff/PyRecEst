"""Systematic conversion between distribution representations.

The conversion layer is intentionally target-centric: a concrete target
representation can implement ``from_distribution(...)`` and the generic
``convert_distribution`` gateway will route to it. This keeps manifold-specific
logic close to the representation that owns it while giving users a single,
discoverable API.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class ConversionError(ValueError):
    """Raised when a distribution representation conversion is unavailable."""


@dataclass
class ConversionResult:
    """Distribution conversion result and metadata.

    Attributes
    ----------
    distribution
        Converted distribution instance.
    source_type
        Concrete source distribution type.
    target_type
        Requested target distribution type.
    method
        Human-readable conversion method used by the gateway.
    exact
        Whether the conversion is mathematically exact as a distribution
        representation. Moment matching, sampling, and grid approximation
        should be reported as approximate.
    parameters
        Conversion parameters that were passed to the conversion routine.
    """

    distribution: Any
    source_type: type
    target_type: type
    method: str
    exact: bool
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class _RegisteredConversion:
    source_type: type
    target_type: type
    converter: Callable[..., Any]
    exact: bool
    method: str


@dataclass
class _ConversionAlias:
    target_type_or_resolver: type | Callable[[Any], type]
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    description: str | None = None


_CONVERSION_REGISTRY: list[_RegisteredConversion] = []
_CONVERSION_ALIAS_REGISTRY: dict[str, _ConversionAlias] = {}


def register_conversion(
    source_type: type,
    target_type: type,
    converter: Callable[..., Any] | None = None,
    *,
    exact: bool = False,
    method: str | None = None,
):
    """Register a custom conversion between two representation classes.

    Parameters
    ----------
    source_type
        Source type accepted by the converter. Subclasses are accepted.
    target_type
        Target type produced by the converter. Subclasses of this target may
        also use the registered conversion.
    converter
        Callable with signature ``converter(source, **kwargs)``. When omitted,
        this function acts as a decorator.
    exact
        Whether the conversion is exact.
    method
        Optional human-readable method name used in :class:`ConversionResult`.

    Returns
    -------
    Callable
        The original converter, enabling decorator use.
    """

    def _decorator(func: Callable[..., Any]):
        _CONVERSION_REGISTRY.insert(
            0,
            _RegisteredConversion(
                source_type=source_type,
                target_type=target_type,
                converter=func,
                exact=exact,
                method=method or _callable_name(func),
            ),
        )
        return func

    if converter is not None:
        return _decorator(converter)
    return _decorator


def register_conversion_alias(
    alias: str,
    target_type_or_resolver: type | Callable[[Any], type],
    *,
    default_kwargs: dict[str, Any] | None = None,
    description: str | None = None,
) -> None:
    """Register a string alias for a target representation.

    Parameters
    ----------
    alias
        Alias accepted by :func:`convert_distribution`, for example
        ``"particles"`` or ``"gaussian"``. Aliases are case-insensitive;
        hyphens and spaces are normalized to underscores.
    target_type_or_resolver
        Concrete target class or callable ``resolver(source_distribution)``
        returning a concrete target class. Resolvers make aliases domain-aware.
    default_kwargs
        Optional conversion arguments supplied by default. User-provided
        keyword arguments override these defaults.
    description
        Optional human-readable description used by
        :func:`registered_conversion_aliases`.
    """

    _CONVERSION_ALIAS_REGISTRY[_normalize_alias(alias)] = _ConversionAlias(
        target_type_or_resolver=target_type_or_resolver,
        default_kwargs=dict(default_kwargs or {}),
        description=description,
    )


def registered_conversion_aliases() -> tuple[tuple[str, str | None], ...]:
    """Return registered custom conversion aliases."""

    return tuple(
        (alias, entry.description) for alias, entry in _CONVERSION_ALIAS_REGISTRY.items()
    )


def convert_distribution(
    distribution,
    target_type,
    /,
    *,
    return_info: bool = False,
    copy_if_same: bool = False,
    **kwargs,
):
    """Convert or approximate ``distribution`` as ``target_type``.

    The gateway supports three cases, in order:

    1. identity conversion when ``distribution`` already is an instance of
       ``target_type``;
    2. explicitly registered conversions;
    3. target classes exposing ``from_distribution(distribution, ...)``.

    Parameters
    ----------
    distribution
        Source distribution instance.
    target_type
        Concrete target representation class, such as
        ``LinearDiracDistribution``, ``CircularGridDistribution``, or
        ``GaussianDistribution``; or a conversion alias such as ``"particles"``,
        ``"gaussian"``, ``"grid"``, or ``"fourier"``.
    return_info
        If false, return the converted distribution directly. If true, return a
        :class:`ConversionResult`.
    copy_if_same
        If true, identity conversion returns a deep copy instead of the same
        object.
    **kwargs
        Parameters required by the target conversion.

    Returns
    -------
    object or ConversionResult
        Converted distribution, or metadata wrapper if ``return_info=True``.
    """

    target_type, alias_kwargs, target_alias = _resolve_target_type(
        distribution, target_type
    )
    conversion_kwargs = {**alias_kwargs, **kwargs}

    if not isinstance(target_type, type):
        raise TypeError("target_type must be a distribution class or conversion alias.")

    source_type = type(distribution)
    method_prefix = f"alias:{target_alias}->" if target_alias is not None else ""

    if isinstance(distribution, target_type):
        converted = copy.deepcopy(distribution) if copy_if_same else distribution
        result = ConversionResult(
            distribution=converted,
            source_type=source_type,
            target_type=target_type,
            method=f"{method_prefix}identity",
            exact=True,
            parameters=dict(conversion_kwargs),
        )
        return result if return_info else result.distribution

    for entry in _matching_registered_conversions(distribution, target_type):
        converted = _call_conversion_callable(
            entry.converter, distribution, conversion_kwargs
        )
        result = ConversionResult(
            distribution=converted,
            source_type=source_type,
            target_type=target_type,
            method=f"{method_prefix}{entry.method}",
            exact=entry.exact,
            parameters=dict(conversion_kwargs),
        )
        return result if return_info else result.distribution

    from_distribution = getattr(target_type, "from_distribution", None)
    if callable(from_distribution):
        converted = _call_conversion_callable(
            from_distribution,
            distribution,
            conversion_kwargs,
            conversion_name=f"{target_type.__name__}.from_distribution",
        )
        result = ConversionResult(
            distribution=converted,
            source_type=source_type,
            target_type=target_type,
            method=f"{method_prefix}{target_type.__name__}.from_distribution",
            exact=False,
            parameters=dict(conversion_kwargs),
        )
        return result if return_info else result.distribution

    raise ConversionError(
        f"No conversion registered for {source_type.__name__} -> {target_type.__name__}. "
        f"Implement {target_type.__name__}.from_distribution(...) or register one with register_conversion(...)."
    )


def can_convert(distribution, target_type) -> bool:
    """Return whether a conversion route exists.

    This only checks for a route. It does not verify that all conversion
    arguments required by the route have been supplied.
    """

    try:
        target_type, _, _ = _resolve_target_type(distribution, target_type)
    except ConversionError:
        return False

    if not isinstance(target_type, type):
        return False
    if isinstance(distribution, target_type):
        return True
    if any(_matching_registered_conversions(distribution, target_type)):
        return True
    return callable(getattr(target_type, "from_distribution", None))


def registered_conversions() -> tuple[tuple[type, type, str, bool], ...]:
    """Return a compact snapshot of explicitly registered conversions."""

    return tuple(
        (entry.source_type, entry.target_type, entry.method, entry.exact)
        for entry in _CONVERSION_REGISTRY
    )


def _normalize_alias(alias: str) -> str:
    return alias.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_target_type(distribution, target_type):
    if isinstance(target_type, str):
        return _resolve_conversion_alias(distribution, target_type)
    return target_type, {}, None


def _resolve_conversion_alias(distribution, alias: str):
    alias_normalized = _normalize_alias(alias)
    custom_alias = _CONVERSION_ALIAS_REGISTRY.get(alias_normalized)
    if custom_alias is not None:
        target_type = _resolve_alias_entry(distribution, custom_alias)
        return target_type, custom_alias.default_kwargs, alias_normalized

    target_type = _resolve_builtin_alias(distribution, alias_normalized)
    return target_type, {}, alias_normalized


def _resolve_alias_entry(distribution, alias_entry: _ConversionAlias):
    target_type_or_resolver = alias_entry.target_type_or_resolver
    if isinstance(target_type_or_resolver, type):
        return target_type_or_resolver
    target_type = target_type_or_resolver(distribution)
    if not isinstance(target_type, type):
        raise ConversionError(
            "Conversion alias resolver must return a distribution class."
        )
    return target_type


def _resolve_builtin_alias(distribution, alias: str):
    # Imports stay local to avoid import cycles with pyrecest.distributions.
    from .circle.abstract_circular_distribution import AbstractCircularDistribution
    from .circle.circular_dirac_distribution import CircularDiracDistribution
    from .circle.circular_fourier_distribution import CircularFourierDistribution
    from .circle.circular_grid_distribution import CircularGridDistribution
    from .hypersphere_subset.abstract_hyperhemispherical_distribution import (
        AbstractHyperhemisphericalDistribution,
    )
    from .hypersphere_subset.abstract_hyperspherical_distribution import (
        AbstractHypersphericalDistribution,
    )
    from .hypersphere_subset.hyperhemispherical_dirac_distribution import (
        HyperhemisphericalDiracDistribution,
    )
    from .hypersphere_subset.hyperhemispherical_grid_distribution import (
        HyperhemisphericalGridDistribution,
    )
    from .hypersphere_subset.hyperspherical_dirac_distribution import (
        HypersphericalDiracDistribution,
    )
    from .hypersphere_subset.hyperspherical_grid_distribution import (
        HypersphericalGridDistribution,
    )
    from .hypertorus.abstract_hypertoroidal_distribution import (
        AbstractHypertoroidalDistribution,
    )
    from .hypertorus.hypertoroidal_dirac_distribution import (
        HypertoroidalDiracDistribution,
    )
    from .hypertorus.hypertoroidal_fourier_distribution import (
        HypertoroidalFourierDistribution,
    )
    from .hypertorus.hypertoroidal_grid_distribution import (
        HypertoroidalGridDistribution,
    )
    from .nonperiodic.abstract_linear_distribution import AbstractLinearDistribution
    from .nonperiodic.gaussian_distribution import GaussianDistribution
    from .nonperiodic.linear_dirac_distribution import LinearDiracDistribution
    from .so3_dirac_distribution import SO3DiracDistribution
    from .so3_product_dirac_distribution import SO3ProductDiracDistribution
    from .so3_product_tangent_gaussian_distribution import (
        SO3ProductTangentGaussianDistribution,
    )
    from .so3_tangent_gaussian_distribution import SO3TangentGaussianDistribution

    if alias in ("gaussian", "normal", "moment_matched_gaussian"):
        if isinstance(distribution, (SO3DiracDistribution, SO3TangentGaussianDistribution)):
            return SO3TangentGaussianDistribution
        if isinstance(
            distribution,
            (SO3ProductDiracDistribution, SO3ProductTangentGaussianDistribution),
        ):
            return SO3ProductTangentGaussianDistribution
        return GaussianDistribution

    if alias in ("linear_dirac", "linear_particles"):
        return LinearDiracDistribution
    if alias in ("so3_dirac", "so3_particles"):
        return SO3DiracDistribution
    if alias in ("so3_product_dirac", "so3_product_particles"):
        return SO3ProductDiracDistribution
    if alias in ("so3_tangent_gaussian", "so3_gaussian"):
        return SO3TangentGaussianDistribution
    if alias in ("so3_product_tangent_gaussian", "so3_product_gaussian"):
        return SO3ProductTangentGaussianDistribution
    if alias == "circular_dirac":
        return CircularDiracDistribution
    if alias == "hypertoroidal_dirac":
        return HypertoroidalDiracDistribution
    if alias == "hyperspherical_dirac":
        return HypersphericalDiracDistribution
    if alias == "hyperhemispherical_dirac":
        return HyperhemisphericalDiracDistribution

    if alias in ("particles", "dirac", "samples"):
        if isinstance(distribution, SO3DiracDistribution):
            return SO3DiracDistribution
        if isinstance(distribution, SO3ProductDiracDistribution):
            return SO3ProductDiracDistribution
        if isinstance(distribution, SO3TangentGaussianDistribution):
            return SO3DiracDistribution
        if isinstance(distribution, SO3ProductTangentGaussianDistribution):
            return SO3ProductDiracDistribution
        if isinstance(distribution, AbstractCircularDistribution):
            return CircularDiracDistribution
        if isinstance(distribution, AbstractHypertoroidalDistribution):
            return HypertoroidalDiracDistribution
        if isinstance(distribution, AbstractHyperhemisphericalDistribution):
            return HyperhemisphericalDiracDistribution
        if isinstance(distribution, AbstractHypersphericalDistribution):
            return HypersphericalDiracDistribution
        if isinstance(distribution, AbstractLinearDistribution):
            return LinearDiracDistribution

    if alias == "circular_grid":
        return CircularGridDistribution
    if alias == "hypertoroidal_grid":
        return HypertoroidalGridDistribution
    if alias == "hyperspherical_grid":
        return HypersphericalGridDistribution
    if alias == "hyperhemispherical_grid":
        return HyperhemisphericalGridDistribution

    if alias == "grid":
        if isinstance(distribution, AbstractCircularDistribution):
            return CircularGridDistribution
        if isinstance(distribution, AbstractHypertoroidalDistribution):
            return HypertoroidalGridDistribution
        if isinstance(distribution, AbstractHyperhemisphericalDistribution):
            return HyperhemisphericalGridDistribution
        if isinstance(distribution, AbstractHypersphericalDistribution):
            return HypersphericalGridDistribution

    if alias == "circular_fourier":
        return CircularFourierDistribution
    if alias == "hypertoroidal_fourier":
        return HypertoroidalFourierDistribution
    if alias == "fourier":
        if isinstance(distribution, AbstractCircularDistribution):
            return CircularFourierDistribution
        if isinstance(distribution, AbstractHypertoroidalDistribution):
            return HypertoroidalFourierDistribution

    raise ConversionError(
        f"Unknown conversion alias '{alias}'. Use a concrete target class or register the alias with register_conversion_alias(...)."
    )


def _matching_registered_conversions(distribution, target_type: type):
    return (
        entry
        for entry in _CONVERSION_REGISTRY
        if isinstance(distribution, entry.source_type)
        and issubclass(target_type, entry.target_type)
    )


def _callable_name(func: Callable[..., Any]) -> str:
    name = getattr(func, "__name__", None)
    if isinstance(name, str):
        return name
    return repr(func)


def _call_conversion_callable(
    func: Callable[..., Any],
    distribution,
    kwargs: dict[str, Any],
    *,
    conversion_name: str | None = None,
):
    name = conversion_name if conversion_name is not None else _callable_name(func)
    _validate_conversion_arguments(func, kwargs, name)
    try:
        return func(distribution, **kwargs)
    except TypeError as exc:
        raise ConversionError(f"Could not call {name}: {exc}") from exc


def _validate_conversion_arguments(
    func: Callable[..., Any], kwargs: dict[str, Any], conversion_name: str
) -> None:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return

    params = list(signature.parameters.values())
    has_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params
    )
    accepted_kwargs: set[str] = set()
    required_kwargs: set[str] = set()
    skipped_source_argument = False

    for param in params:
        if (
            not skipped_source_argument
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ):
            skipped_source_argument = True
            continue

        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted_kwargs.add(param.name)
            if param.default is inspect.Parameter.empty:
                required_kwargs.add(param.name)

    missing = required_kwargs - set(kwargs)
    if missing:
        raise ConversionError(
            f"{conversion_name} is missing required conversion argument(s): "
            f"{', '.join(sorted(missing))}."
        )

    if has_var_keyword:
        return

    unknown = set(kwargs) - accepted_kwargs
    if unknown:
        accepted = ", ".join(sorted(accepted_kwargs)) or "no keyword arguments"
        raise ConversionError(
            f"{conversion_name} got unsupported conversion argument(s): "
            f"{', '.join(sorted(unknown))}. Accepted arguments: {accepted}."
        )


from . import so3_conversion as _so3_conversion  # noqa: F401,E402  pylint: disable=wrong-import-position,unused-import


__all__ = [
    "ConversionError",
    "ConversionResult",
    "can_convert",
    "convert_distribution",
    "register_conversion",
    "register_conversion_alias",
    "registered_conversion_aliases",
    "registered_conversions",
]
