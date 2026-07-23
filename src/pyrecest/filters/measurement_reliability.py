"""Helpers for reliability-weighted measurement batches."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from pyrecest.backend import all, array, isfinite, ones, reshape, stack, zeros


@dataclass(frozen=True)
class MeasurementReliabilitySelection:
    """Normalized reliability information for a measurement batch.

    ``measurement_weights`` is a backend array with one non-negative finite
    weight per measurement. ``active_measurement_mask`` is a Python bool list so
    it can be used without backend-specific truth-value ambiguity.
    ``active_measurement_indices`` contains measurements that are both active
    and have strictly positive weight.
    """

    measurement_weights: Any
    active_measurement_mask: list[bool]
    active_measurement_indices: list[int]


def _normalize_integer_count(
    value: Any, name: str, *, minimum: int, message: str
) -> int:
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(message)

    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(scalar, (int, np.integer)):
        parsed = int(scalar)
    elif (
        isinstance(scalar, (float, np.floating))
        and np.isfinite(scalar)
        and float(scalar).is_integer()
    ):
        parsed = int(scalar)
    else:
        raise ValueError(message)

    if parsed < minimum:
        raise ValueError(message)
    return parsed


def _normalize_nonnegative_integer(value: Any, name: str) -> int:
    return _normalize_integer_count(
        value,
        name,
        minimum=0,
        message=f"{name} must be a non-negative integer",
    )


def _normalize_positive_integer(value: Any, name: str) -> int:
    return _normalize_integer_count(
        value,
        name,
        minimum=1,
        message=f"{name} must be a positive integer",
    )


def _is_boolean_object_value(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    kind = getattr(dtype, "kind", None)
    if kind is not None:
        return kind == "b"
    return "bool" in str(dtype).lower()


def _object_array_contains_only_boolean_values(value) -> bool:
    try:
        flat_values = reshape(value, (-1,))
    except (TypeError, ValueError, AttributeError):
        item = value.item() if hasattr(value, "item") else value
        return _is_boolean_object_value(item)
    return builtins.all(
        _is_boolean_object_value(flat_values[index])
        for index in range(flat_values.shape[0])
    )


def _has_boolean_dtype(value) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    kind = getattr(dtype, "kind", None)
    if kind == "b":
        return True
    if kind == "O" or (kind is None and "object" in str(dtype).lower()):
        return _object_array_contains_only_boolean_values(value)
    return "bool" in str(dtype).lower()


def _has_object_dtype(value) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    kind = getattr(dtype, "kind", None)
    return kind == "O" or (kind is None and "object" in str(dtype).lower())


def _is_real_numeric_object_value(value) -> bool:
    if isinstance(value, (bool, complex, str, bytes, bytearray)):
        return False
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        kind = getattr(dtype, "kind", None)
        if kind is not None:
            return kind in {"i", "u", "f"}
        dtype_name = str(dtype).lower()
        if any(
            token in dtype_name
            for token in ("bool", "complex", "str", "string", "object")
        ):
            return False
        if "float" in dtype_name or "int" in dtype_name:
            return True
    try:
        float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _object_array_contains_only_real_numeric_values(value) -> bool:
    try:
        flat_values = reshape(value, (-1,))
    except (TypeError, ValueError, AttributeError):
        item = value.item() if hasattr(value, "item") else value
        return _is_real_numeric_object_value(item)
    return builtins.all(
        _is_real_numeric_object_value(flat_values[index])
        for index in range(flat_values.shape[0])
    )


def _has_real_numeric_dtype(value) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    if _has_object_dtype(value):
        return _object_array_contains_only_real_numeric_values(value)
    kind = getattr(dtype, "kind", None)
    if kind is not None:
        return kind in {"i", "u", "f"}
    dtype_name = str(dtype).lower()
    if any(
        token in dtype_name for token in ("bool", "complex", "str", "string", "object")
    ):
        return False
    return "float" in dtype_name or "int" in dtype_name


def _coerce_object_weight_array_to_float(weights):
    if not _has_object_dtype(weights):
        return weights
    flat_weights = reshape(weights, (-1,))
    return array([float(flat_weights[index]) for index in range(flat_weights.shape[0])])


def _raise_if_not_real_numeric_weights(weights) -> None:
    if not _has_real_numeric_dtype(weights):
        raise ValueError("measurement_weights must be real numeric")


def normalize_measurement_weights(measurement_weights, n_measurements: int):
    """Return one non-negative finite reliability weight per measurement.

    ``measurement_weights`` may be ``None`` (all ones), a scalar, or a vector
    with one entry per measurement. Zero weights are valid and typically mean
    that the corresponding measurement should be skipped.
    """

    n_measurements = _normalize_nonnegative_integer(n_measurements, "n_measurements")
    if measurement_weights is None:
        return ones(n_measurements)

    weights = array(measurement_weights)
    _raise_if_not_real_numeric_weights(weights)
    if weights.ndim == 0:
        try:
            scalar_weight = float(weights)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("measurement_weights must be real numeric") from exc
        if not np.isfinite(scalar_weight):
            raise ValueError("measurement_weights must be finite")
        if scalar_weight < 0.0:
            raise ValueError("measurement_weights must be non-negative")
        weights = ones(n_measurements) * scalar_weight
    else:
        weights = reshape(weights, (-1,))
        if weights.shape[0] != n_measurements:
            raise ValueError(
                "measurement_weights must be scalar or have one entry per measurement"
            )
        weights = _coerce_object_weight_array_to_float(weights)
    try:
        weights_are_finite = bool(all(isfinite(weights)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("measurement_weights must be real numeric") from exc
    if not weights_are_finite:
        raise ValueError("measurement_weights must be finite")
    try:
        weights_are_nonnegative = bool(all(weights >= 0.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("measurement_weights must be real numeric") from exc
    if not weights_are_nonnegative:
        raise ValueError("measurement_weights must be non-negative")
    return weights


def normalize_active_measurement_mask(
    active_measurement_mask,
    n_measurements: int,
) -> list[bool]:
    """Return one boolean active/inactive flag per measurement."""

    n_measurements = _normalize_nonnegative_integer(n_measurements, "n_measurements")
    if active_measurement_mask is None:
        return [True] * n_measurements

    mask = array(active_measurement_mask)
    if not _has_boolean_dtype(mask):
        raise ValueError("active_measurement_mask must contain booleans")
    if mask.ndim == 0:
        return [bool(mask)] * n_measurements
    mask = reshape(mask, (-1,))
    if mask.shape[0] != n_measurements:
        raise ValueError(
            "active_measurement_mask must be scalar or have one entry per measurement"
        )
    return [bool(mask[index]) for index in range(n_measurements)]


def normalize_measurement_reliability(
    measurement_weights,
    active_measurement_mask,
    n_measurements: int,
) -> MeasurementReliabilitySelection:
    """Normalize measurement weights and active-mask inputs together."""

    n_measurements = _normalize_nonnegative_integer(n_measurements, "n_measurements")
    weights = normalize_measurement_weights(measurement_weights, n_measurements)
    mask = normalize_active_measurement_mask(active_measurement_mask, n_measurements)
    active_indices = [
        index
        for index, is_active in enumerate(mask)
        if is_active and float(weights[index]) > 0.0
    ]
    return MeasurementReliabilitySelection(
        measurement_weights=weights,
        active_measurement_mask=mask,
        active_measurement_indices=active_indices,
    )


def normalize_measurement_noise_covariances(
    measurement_noise,
    n_measurements: int,
    measurement_dim: int,
    *,
    as_covariance_matrix: Callable[[Any, int, str], Any],
    name: str = "R",
):
    """Return one covariance matrix per measurement.

    ``measurement_noise`` may be a shared covariance specification accepted by
    ``as_covariance_matrix`` or an array with shape
    ``(n_measurements, measurement_dim, measurement_dim)``. The callback lets
    tracker classes reuse their own covariance validation conventions.
    """

    n_measurements = _normalize_nonnegative_integer(n_measurements, "n_measurements")
    measurement_dim = _normalize_positive_integer(measurement_dim, "measurement_dim")

    noise = array(measurement_noise)
    empty_shape = (0, measurement_dim, measurement_dim)
    if noise.ndim == 3:
        expected_shape = (n_measurements, measurement_dim, measurement_dim)
        if noise.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape ({measurement_dim}, {measurement_dim}) "
                f"or ({n_measurements}, {measurement_dim}, {measurement_dim}) "
                "for per-measurement covariances"
            )
        if n_measurements == 0:
            return zeros(empty_shape)
        return stack(
            [
                as_covariance_matrix(
                    noise[index],
                    measurement_dim,
                    f"{name}[{index}]",
                )
                for index in range(n_measurements)
            ]
        )

    shared_noise = as_covariance_matrix(noise, measurement_dim, name)
    if n_measurements == 0:
        return zeros(empty_shape)
    return stack([shared_noise for _ in range(n_measurements)])
