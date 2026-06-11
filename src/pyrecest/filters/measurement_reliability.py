"""Helpers for reliability-weighted measurement batches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


def normalize_measurement_weights(measurement_weights, n_measurements: int):
    """Return one non-negative finite reliability weight per measurement.

    ``measurement_weights`` may be ``None`` (all ones), a scalar, or a vector
    with one entry per measurement. Zero weights are valid and typically mean
    that the corresponding measurement should be skipped.
    """

    if n_measurements < 0:
        raise ValueError("n_measurements must be non-negative")
    if measurement_weights is None:
        return ones(n_measurements)

    weights = array(measurement_weights)
    if weights.ndim == 0:
        weights = ones(n_measurements) * float(weights)
    else:
        weights = reshape(weights, (-1,))
        if weights.shape[0] != n_measurements:
            raise ValueError(
                "measurement_weights must be scalar or have one entry per measurement"
            )
    if not bool(all(isfinite(weights))):
        raise ValueError("measurement_weights must be finite")
    if not bool(all(weights >= 0.0)):
        raise ValueError("measurement_weights must be non-negative")
    return weights


def normalize_active_measurement_mask(
    active_measurement_mask,
    n_measurements: int,
) -> list[bool]:
    """Return one boolean active/inactive flag per measurement."""

    if n_measurements < 0:
        raise ValueError("n_measurements must be non-negative")
    if active_measurement_mask is None:
        return [True] * n_measurements

    mask = array(active_measurement_mask)
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

    if n_measurements < 0:
        raise ValueError("n_measurements must be non-negative")
    if measurement_dim <= 0:
        raise ValueError("measurement_dim must be positive")

    if n_measurements == 0:
        return zeros((0, measurement_dim, measurement_dim))

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
