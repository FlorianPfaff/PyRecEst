"""Protocol-compliance testing helpers for PyRecEst extension points.

The helpers in this module validate lightweight interface contracts. They are
intended for tests of built-in and user-defined PyRecEst components, not for
mathematical correctness proofs. Protocol-specific tests should still check the
actual numerical semantics of each implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Integral
from typing import Any, TypeVar, cast

from .common import SupportsDim, SupportsInputDim

T = TypeVar("T")

_MISSING = object()


class ProtocolAssertionError(AssertionError):
    """Assertion error raised by protocol-compliance test helpers."""


def _type_name(obj: object) -> str:
    return type(obj).__name__


def _normalise_shape(shape: Iterable[int], value_name: str) -> tuple[int, ...]:
    try:
        return tuple(int(axis) for axis in shape)
    except TypeError as exc:
        raise ProtocolAssertionError(f"{value_name} must be an iterable shape, got {shape!r}.") from exc


def _shape_of(value: object, value_name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", _MISSING)
    if shape is _MISSING:
        raise ProtocolAssertionError(f"{value_name} must expose a 'shape' attribute.")
    return _normalise_shape(cast(Iterable[int], shape), value_name)


def _nonnegative_integer(value: object, value_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ProtocolAssertionError(f"{value_name} must be an integer, got {value!r}.")

    int_value = int(value)
    if int_value < 0:
        raise ProtocolAssertionError(f"{value_name} must be non-negative, got {int_value}.")
    return int_value


def assert_protocol_instance(obj: object, protocol: type[Any], *, protocol_name: str | None = None) -> None:
    """Assert that ``obj`` satisfies a runtime-checkable protocol.

    Parameters
    ----------
    obj
        Object to check.
    protocol
        Runtime-checkable protocol or other class accepted by :func:`isinstance`.
    protocol_name
        Optional display name used in failure messages.
    """

    display_name = protocol_name or getattr(protocol, "__name__", repr(protocol))
    try:
        conforms = isinstance(obj, protocol)
    except TypeError as exc:
        raise ProtocolAssertionError(
            f"{display_name} cannot be used for runtime checks. Decorate protocol classes with typing.runtime_checkable before using this helper."
        ) from exc

    if not conforms:
        raise ProtocolAssertionError(f"{_type_name(obj)} does not satisfy {display_name}.")


def assert_has_attribute(obj: object, attribute_name: str) -> Any:
    """Return an attribute and fail clearly if it is missing."""

    value = getattr(obj, attribute_name, _MISSING)
    if value is _MISSING:
        raise ProtocolAssertionError(f"{_type_name(obj)} must provide attribute {attribute_name!r}.")
    return value


def assert_callable_attribute(obj: object, attribute_name: str) -> Callable[..., Any]:
    """Return a callable attribute and fail clearly if it is missing or not callable."""

    value = assert_has_attribute(obj, attribute_name)
    if not callable(value):
        raise ProtocolAssertionError(f"{_type_name(obj)}.{attribute_name} must be callable.")
    return cast(Callable[..., Any], value)


def assert_value_is_not_none(value: T | None, *, value_name: str = "value") -> T:
    """Return ``value`` and fail clearly if it is ``None``."""

    if value is None:
        raise ProtocolAssertionError(f"{value_name} must not be None.")
    return value


def assert_method_returns_non_none(obj: object, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call a method and assert that its result is not ``None``."""

    method = assert_callable_attribute(obj, method_name)
    return assert_value_is_not_none(method(*args, **kwargs), value_name=f"{_type_name(obj)}.{method_name}(...) result")


def assert_shape(value: object, expected_shape: Iterable[int], *, value_name: str = "value") -> tuple[int, ...]:
    """Assert that an array-like value exposes exactly ``expected_shape``."""

    actual_shape = _shape_of(value, value_name)
    normalised_expected_shape = _normalise_shape(expected_shape, "expected_shape")
    if actual_shape != normalised_expected_shape:
        raise ProtocolAssertionError(f"{value_name} must have shape {normalised_expected_shape}, got {actual_shape}.")
    return actual_shape


def assert_shape_prefix(value: object, expected_prefix: Iterable[int], *, value_name: str = "value") -> tuple[int, ...]:
    """Assert that an array-like value's shape starts with ``expected_prefix``."""

    actual_shape = _shape_of(value, value_name)
    normalised_expected_prefix = _normalise_shape(expected_prefix, "expected_prefix")
    if actual_shape[: len(normalised_expected_prefix)] != normalised_expected_prefix:
        raise ProtocolAssertionError(f"{value_name} shape must start with {normalised_expected_prefix}, got {actual_shape}.")
    return actual_shape


def assert_trailing_dimension(value: object, expected_dim: int, *, value_name: str = "value") -> tuple[int, ...]:
    """Assert that an array-like value has ``expected_dim`` on the final axis."""

    actual_shape = _shape_of(value, value_name)
    normalised_expected_dim = _nonnegative_integer(expected_dim, "expected_dim")
    if not actual_shape:
        raise ProtocolAssertionError(f"{value_name} must have at least one axis, got scalar shape ().")
    if actual_shape[-1] != normalised_expected_dim:
        raise ProtocolAssertionError(f"{value_name} trailing dimension must be {normalised_expected_dim}, got shape {actual_shape}.")
    return actual_shape


def assert_supports_dim(obj: object) -> int:
    """Assert that an object satisfies :class:`SupportsDim` and return ``dim``."""

    assert_protocol_instance(obj, SupportsDim)
    return _nonnegative_integer(assert_has_attribute(obj, "dim"), "dim")


def assert_supports_input_dim(obj: object) -> int:
    """Assert that an object satisfies :class:`SupportsInputDim` and return ``input_dim``."""

    assert_protocol_instance(obj, SupportsInputDim)
    return _nonnegative_integer(assert_has_attribute(obj, "input_dim"), "input_dim")


def assert_supports_pdf(distribution: object, xs: Any) -> Any:
    """Assert that ``distribution.pdf(xs)`` exists and returns a non-``None`` value."""

    return assert_method_returns_non_none(distribution, "pdf", xs)


def assert_supports_ln_pdf(distribution: object, xs: Any) -> Any:
    """Assert that ``distribution.ln_pdf(xs)`` exists and returns a non-``None`` value."""

    return assert_method_returns_non_none(distribution, "ln_pdf", xs)


def assert_supports_sampling(distribution: object, n: int = 5) -> Any:
    """Assert that ``distribution.sample(n)`` exists and returns a non-``None`` value."""

    _nonnegative_integer(n, "n")
    return assert_method_returns_non_none(distribution, "sample", n)


def assert_supports_mean(distribution: object) -> Any:
    """Assert that ``distribution.mean()`` exists and returns a non-``None`` value."""

    return assert_method_returns_non_none(distribution, "mean")


def assert_supports_covariance(distribution: object) -> Any:
    """Assert that ``distribution.covariance()`` exists and returns a non-``None`` value."""

    return assert_method_returns_non_none(distribution, "covariance")


def assert_filter_basic_contract(filter_obj: object) -> Any:
    """Assert the minimal PyRecEst recursive-filter test contract.

    The checked contract is deliberately small: a filter exposes ``dim``, a
    ``filter_state`` attribute, and a non-``None`` ``get_point_estimate()``
    result. More specific filter helpers can be added once filter capability
    protocols exist.
    """

    assert_supports_dim(filter_obj)
    assert_has_attribute(filter_obj, "filter_state")
    return assert_method_returns_non_none(filter_obj, "get_point_estimate")


def assert_supports_likelihood(model: object, measurement: Any, state: Any) -> Any:
    """Assert that ``model.likelihood(measurement, state)`` returns a value."""

    return assert_method_returns_non_none(model, "likelihood", measurement, state)


def assert_supports_log_likelihood(model: object, measurement: Any, state: Any) -> Any:
    """Assert that ``model.log_likelihood(measurement, state)`` returns a value."""

    return assert_method_returns_non_none(model, "log_likelihood", measurement, state)


def assert_supports_transition_sampling(model: object, state: Any, n: int = 1) -> Any:
    """Assert that ``model.sample_next(state, n)`` returns next-state samples."""

    _nonnegative_integer(n, "n")
    return assert_method_returns_non_none(model, "sample_next", state, n)


def assert_supports_transition_density(model: object, state_next: Any, state_previous: Any) -> Any:
    """Assert that ``model.transition_density(state_next, state_previous)`` returns a value."""

    return assert_method_returns_non_none(model, "transition_density", state_next, state_previous)


__all__ = [
    "ProtocolAssertionError",
    "assert_callable_attribute",
    "assert_filter_basic_contract",
    "assert_has_attribute",
    "assert_method_returns_non_none",
    "assert_protocol_instance",
    "assert_shape",
    "assert_shape_prefix",
    "assert_supports_covariance",
    "assert_supports_dim",
    "assert_supports_input_dim",
    "assert_supports_likelihood",
    "assert_supports_ln_pdf",
    "assert_supports_log_likelihood",
    "assert_supports_mean",
    "assert_supports_pdf",
    "assert_supports_sampling",
    "assert_supports_transition_density",
    "assert_supports_transition_sampling",
    "assert_trailing_dimension",
    "assert_value_is_not_none",
]
