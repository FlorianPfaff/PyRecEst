"""Small protocol-compliance assertion helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Integral
from typing import Any, TypeVar, cast

from .common import SupportsDim, SupportsInputDim

T = TypeVar("T")
_MISSING = object()


class ProtocolAssertionError(AssertionError):
    """Raised when a protocol helper detects a contract mismatch."""


def _type_name(obj: object) -> str:
    return type(obj).__name__


def _normalise_shape(shape: Iterable[int], value_name: str) -> tuple[int, ...]:
    try:
        return tuple(int(axis) for axis in shape)
    except TypeError as exc:
        raise ProtocolAssertionError(
            f"{value_name} must be an iterable shape."
        ) from exc


def _shape_of(value: object, value_name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", _MISSING)
    if shape is _MISSING:
        raise ProtocolAssertionError(f"{value_name} must expose a shape attribute.")
    return _normalise_shape(cast(Iterable[int], shape), value_name)


def _nonnegative_integer(value: object, value_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ProtocolAssertionError(f"{value_name} must be an integer.")
    int_value = int(value)
    if int_value < 0:
        raise ProtocolAssertionError(f"{value_name} must be non-negative.")
    return int_value


def assert_protocol_instance(
    obj: object, protocol: type[Any], *, protocol_name: str | None = None
) -> None:
    name = protocol_name or getattr(protocol, "__name__", repr(protocol))
    try:
        conforms = isinstance(obj, protocol)
    except TypeError as exc:
        raise ProtocolAssertionError(
            f"{name} cannot be used for runtime checks."
        ) from exc
    if not conforms:
        raise ProtocolAssertionError(f"{_type_name(obj)} does not satisfy {name}.")


def assert_has_attribute(obj: object, attribute_name: str) -> Any:
    value = getattr(obj, attribute_name, _MISSING)
    if value is _MISSING:
        raise ProtocolAssertionError(
            f"{_type_name(obj)} must provide {attribute_name!r}."
        )
    return value


def assert_callable_attribute(obj: object, attribute_name: str) -> Callable[..., Any]:
    value = assert_has_attribute(obj, attribute_name)
    if not callable(value):
        raise ProtocolAssertionError(
            f"{_type_name(obj)}.{attribute_name} must be callable."
        )
    return cast(Callable[..., Any], value)


def assert_value_is_not_none(value: T | None, *, value_name: str = "value") -> T:
    if value is None:
        raise ProtocolAssertionError(f"{value_name} must not be None.")
    return value


def assert_method_returns_non_none(
    obj: object, method_name: str, *args: Any, **kwargs: Any
) -> Any:
    method = assert_callable_attribute(obj, method_name)
    return assert_value_is_not_none(
        method(*args, **kwargs), value_name=f"{method_name} result"
    )


def assert_shape(
    value: object, expected_shape: Iterable[int], *, value_name: str = "value"
) -> tuple[int, ...]:
    actual = _shape_of(value, value_name)
    expected = _normalise_shape(expected_shape, "expected_shape")
    if actual != expected:
        raise ProtocolAssertionError(
            f"{value_name} must have shape {expected}, got {actual}."
        )
    return actual


def assert_shape_prefix(
    value: object, expected_prefix: Iterable[int], *, value_name: str = "value"
) -> tuple[int, ...]:
    actual = _shape_of(value, value_name)
    expected = _normalise_shape(expected_prefix, "expected_prefix")
    if actual[: len(expected)] != expected:
        raise ProtocolAssertionError(
            f"{value_name} shape must start with {expected}, got {actual}."
        )
    return actual


def assert_trailing_dimension(
    value: object, expected_dim: int, *, value_name: str = "value"
) -> tuple[int, ...]:
    actual = _shape_of(value, value_name)
    dim = _nonnegative_integer(expected_dim, "expected_dim")
    if not actual or actual[-1] != dim:
        raise ProtocolAssertionError(
            f"{value_name} trailing dimension must be {dim}, got {actual}."
        )
    return actual


def assert_supports_dim(obj: object) -> int:
    assert_protocol_instance(obj, SupportsDim)
    return _nonnegative_integer(assert_has_attribute(obj, "dim"), "dim")


def assert_supports_input_dim(obj: object) -> int:
    assert_protocol_instance(obj, SupportsInputDim)
    return _nonnegative_integer(assert_has_attribute(obj, "input_dim"), "input_dim")


def assert_supports_pdf(distribution: object, xs: Any) -> Any:
    return assert_method_returns_non_none(distribution, "pdf", xs)


def assert_supports_ln_pdf(distribution: object, xs: Any) -> Any:
    return assert_method_returns_non_none(distribution, "ln_pdf", xs)


def assert_supports_sampling(distribution: object, n: int = 5) -> Any:
    _nonnegative_integer(n, "n")
    return assert_method_returns_non_none(distribution, "sample", n)


def assert_supports_mean(distribution: object) -> Any:
    return assert_method_returns_non_none(distribution, "mean")


def assert_supports_covariance(distribution: object) -> Any:
    return assert_method_returns_non_none(distribution, "covariance")


def assert_filter_basic_contract(filter_obj: object) -> Any:
    assert_supports_dim(filter_obj)
    assert_has_attribute(filter_obj, "filter_state")
    return assert_method_returns_non_none(filter_obj, "get_point_estimate")


def assert_supports_likelihood(model: object, measurement: Any, state: Any) -> Any:
    return assert_method_returns_non_none(model, "likelihood", measurement, state)


def assert_supports_log_likelihood(model: object, measurement: Any, state: Any) -> Any:
    return assert_method_returns_non_none(model, "log_likelihood", measurement, state)


def assert_supports_transition_sampling(model: object, state: Any, n: int = 1) -> Any:
    _nonnegative_integer(n, "n")
    return assert_method_returns_non_none(model, "sample_next", state, n)


def assert_supports_transition_density(
    model: object, state_next: Any, state_previous: Any
) -> Any:
    return assert_method_returns_non_none(
        model, "transition_density", state_next, state_previous
    )


__all__ = [name for name in globals() if name.startswith("assert_")] + [
    "ProtocolAssertionError"
]
