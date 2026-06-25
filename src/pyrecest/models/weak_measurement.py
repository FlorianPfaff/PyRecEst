"Utilities for masked and weak-dimension linear Gaussian measurements."

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .linear_gaussian import LinearGaussianMeasurementModel

_BOOL_OR_TEXT_KINDS = {"b", "S", "U"}
_BOOL_OR_TEXT_SCALAR_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
)


def diagonal_measurement_covariance(stds: Sequence[float] | np.ndarray) -> np.ndarray:
    "Return a diagonal covariance matrix from measurement standard deviations."
    values = _standard_deviations_array(stds)
    return np.diag(values**2)


def block_diag_measurement_covariance(
    *,
    trusted_std: Mapping[Any, float] | Sequence[float] | None = None,
    weak_std: Mapping[Any, float] | Sequence[float] | None = None,
    dimension_order: Sequence[Any] | None = None,
) -> np.ndarray:
    "Return a diagonal covariance from trusted and weak dimension stds."
    if trusted_std is None and weak_std is None:
        raise ValueError("at least one of trusted_std or weak_std must be provided")
    if _is_mapping(trusted_std) or _is_mapping(weak_std):
        if trusted_std is not None and not _is_mapping(trusted_std):
            raise TypeError("trusted_std must be a mapping when weak_std is a mapping")
        if weak_std is not None and not _is_mapping(weak_std):
            raise TypeError("weak_std must be a mapping when trusted_std is a mapping")
        trusted_map = dict(trusted_std or {})
        weak_map = dict(weak_std or {})
        provided_order = [
            *trusted_map.keys(),
            *(key for key in weak_map if key not in trusted_map),
        ]
        if dimension_order is None:
            order = provided_order
        else:
            order = list(dimension_order)
            if len(set(order)) != len(order):
                raise ValueError("dimension_order must not contain duplicate entries")
            omitted = [key for key in provided_order if key not in order]
            if omitted:
                raise KeyError(f"dimension_order omits std entries: {omitted}")
        missing = [
            key for key in order if key not in trusted_map and key not in weak_map
        ]
        if missing:
            raise KeyError(f"dimension_order contains missing std entries: {missing}")
        return diagonal_measurement_covariance(
            [trusted_map[key] if key in trusted_map else weak_map[key] for key in order]
        )

    stds: list[Any] = []
    if trusted_std is not None:
        stds.extend(trusted_std)
    if weak_std is not None:
        stds.extend(weak_std)
    return diagonal_measurement_covariance(stds)


def selection_matrix(state_dim: int, observed_dims: Sequence[int]) -> np.ndarray:
    "Return a matrix selecting observed state components."
    state_dim = _positive_int(state_dim, "state_dim")
    dims = [_nonnegative_int(dim, "observed_dims") for dim in observed_dims]
    if not dims:
        raise ValueError("observed_dims must contain at least one state index")
    if len(set(dims)) != len(dims):
        raise ValueError("observed_dims must not contain duplicate indices")
    if any(dim >= state_dim for dim in dims):
        raise ValueError("observed_dims must be valid indices for state_dim")
    matrix = np.zeros((len(dims), state_dim), dtype=float)
    for row, dim in enumerate(dims):
        matrix[row, dim] = 1.0
    return matrix


class MaskedLinearMeasurementModel(LinearGaussianMeasurementModel):
    "Linear Gaussian model that observes a subset of state dimensions."

    def __init__(
        self,
        *,
        state_dim: int,
        observed_dims: Sequence[int],
        measurement_noise_cov: np.ndarray | None = None,
        stds: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if (measurement_noise_cov is None) == (stds is None):
            raise ValueError("provide exactly one of measurement_noise_cov or stds")
        observed_dims_tuple = tuple(observed_dims)
        matrix = selection_matrix(state_dim, observed_dims_tuple)
        covariance = (
            diagonal_measurement_covariance(stds)
            if stds is not None
            else np.asarray(measurement_noise_cov, dtype=float)
        )
        super().__init__(matrix, covariance)
        self.observed_dims = tuple(
            _nonnegative_int(dim, "observed_dims") for dim in observed_dims_tuple
        )


class WeakDimensionMeasurementModel(LinearGaussianMeasurementModel):
    "Linear Gaussian model with per-dimension measurement trust levels."

    def __init__(
        self,
        measurement_matrix: np.ndarray,
        *,
        stds: Mapping[Any, float] | Sequence[float] | np.ndarray | None = None,
        trusted_std: Mapping[Any, float] | Sequence[float] | None = None,
        weak_std: Mapping[Any, float] | Sequence[float] | None = None,
        dimension_order: Sequence[Any] | None = None,
        measurement_noise_cov: np.ndarray | None = None,
    ) -> None:
        provided = sum(
            item is not None
            for item in (stds, measurement_noise_cov, trusted_std, weak_std)
        )
        if provided == 0:
            raise ValueError(
                "provide stds, measurement_noise_cov, or trusted/weak stds"
            )
        if measurement_noise_cov is not None and provided > 1:
            raise ValueError(
                "measurement_noise_cov cannot be combined with std arguments"
            )
        if stds is not None and (trusted_std is not None or weak_std is not None):
            raise ValueError("stds cannot be combined with trusted_std or weak_std")
        if measurement_noise_cov is not None:
            covariance = np.asarray(measurement_noise_cov, dtype=float)
        elif stds is not None and _is_mapping(stds):
            covariance = block_diag_measurement_covariance(
                trusted_std=stds, dimension_order=dimension_order
            )
        elif stds is not None:
            covariance = diagonal_measurement_covariance(stds)
        else:
            covariance = block_diag_measurement_covariance(
                trusted_std=trusted_std,
                weak_std=weak_std,
                dimension_order=dimension_order,
            )
        super().__init__(measurement_matrix, covariance)


def masked_position_measurement_model(
    state_dim: int,
    observed_dims: Sequence[int],
    stds: Sequence[float] | np.ndarray,
) -> MaskedLinearMeasurementModel:
    "Convenience constructor for masked position-like measurements."
    return MaskedLinearMeasurementModel(
        state_dim=state_dim, observed_dims=observed_dims, stds=stds
    )


def weak_dimension_measurement_model(
    measurement_matrix: np.ndarray,
    stds: Sequence[float] | np.ndarray,
) -> WeakDimensionMeasurementModel:
    "Convenience constructor for weak-dimension linear measurements."
    return WeakDimensionMeasurementModel(measurement_matrix, stds=stds)


def _is_mapping(value: object) -> bool:
    return isinstance(value, Mapping)


def _standard_deviations_array(stds: Sequence[float] | np.ndarray) -> np.ndarray:
    try:
        if _contains_bool_or_text_values(stds):
            raise ValueError
        values = np.asarray(stds, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("stds must contain real numeric values") from exc
    if values.size == 0:
        raise ValueError("stds must contain at least one standard deviation")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("stds must contain finite positive values")
    return values


def _contains_bool_or_text_values(value: Any) -> bool:
    array = np.asarray(value)
    if array.dtype.kind in _BOOL_OR_TEXT_KINDS:
        return True
    if array.dtype.kind != "O":
        return False
    return any(isinstance(item, _BOOL_OR_TEXT_SCALAR_TYPES) for item in array.flat)


def _positive_int(value: int, name: str) -> int:
    parsed = _nonnegative_int(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_int(value: int, name: str) -> int:
    try:
        array_value = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if array_value.ndim != 0 or array_value.dtype.kind == "b":
        raise ValueError(f"{name} must be a nonnegative integer")
    scalar = array_value.item()
    if type(scalar) is bool:
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        parsed = int(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed != scalar or parsed < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return parsed


__all__ = [
    "MaskedLinearMeasurementModel",
    "WeakDimensionMeasurementModel",
    "block_diag_measurement_covariance",
    "diagonal_measurement_covariance",
    "masked_position_measurement_model",
    "selection_matrix",
    "weak_dimension_measurement_model",
]
