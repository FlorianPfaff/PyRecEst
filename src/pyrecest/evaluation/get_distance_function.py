from __future__ import annotations

from collections.abc import Callable
from math import pi
from typing import Any

import numpy as np
from pyrecest.backend import arccos, asarray, clip, dot, linalg, to_numpy
from pyrecest.distributions import AbstractHypertoroidalDistribution
from scipy.optimize import linear_sum_assignment

DistanceFactory = Callable[[str, dict[str, Any] | None], Callable[[Any, Any], float]]
_DISTANCE_FUNCTION_FACTORIES: dict[str, DistanceFactory] = {}
_UNSUPPORTED_NUMERIC_CONFIG_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
    complex,
    np.complexfloating,
)


def _normalize_registry_name(manifold_name: str) -> str:
    if not isinstance(manifold_name, str) or not manifold_name.strip():
        raise ValueError("manifold_name must be a non-empty string")
    return manifold_name.strip().lower()


def register_distance_function(
    manifold_name: str, factory: DistanceFactory
) -> DistanceFactory:
    """Register a custom distance-function factory for a manifold name."""
    normalized_name = _normalize_registry_name(manifold_name)
    if not callable(factory):
        raise TypeError("factory must be callable")
    _DISTANCE_FUNCTION_FACTORIES[normalized_name] = factory
    return factory


def available_distance_functions() -> tuple[str, ...]:
    return tuple(sorted(_DISTANCE_FUNCTION_FACTORIES))


def _is_hypersphere_symmetric_name(normalized_name: str) -> bool:
    return (
        "hyperspheresymmetric" in normalized_name
        or "hypersphere_symmetric" in normalized_name
    )


def _without_symmetry_suffix(manifold_name: str) -> str:
    return (
        manifold_name.replace("hypersphereSymmetric", "hypersphere")
        .replace("hypersphere_symmetric", "hypersphere")
        .replace("_symmetric", "")
        .replace("Symmetric", "")
        .replace("symmetric", "")
        .replace("Symm", "")
        .replace("symm", "")
    )


def _contains_unsupported_numeric_config_values(value: Any) -> bool:
    if isinstance(value, _UNSUPPORTED_NUMERIC_CONFIG_TYPES):
        return True
    try:
        raw_values = np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        raw_values = ()
    if any(isinstance(item, _UNSUPPORTED_NUMERIC_CONFIG_TYPES) for item in raw_values):
        return True
    try:
        values = np.asarray(to_numpy(value), dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _UNSUPPORTED_NUMERIC_CONFIG_TYPES) for item in values)


def _as_real_numeric_array(value: Any, name: str) -> np.ndarray:
    message = f"{name} must contain only finite real numeric values"
    if _contains_unsupported_numeric_config_values(value):
        raise ValueError(message)
    try:
        values = np.asarray(to_numpy(value), dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(message)
    return values


def _validate_symmetry_count(nSymm: Any) -> int:
    count_array = np.asarray(to_numpy(nSymm))
    if (
        count_array.shape != ()
        or np.issubdtype(count_array.dtype, np.bool_)
        or _contains_unsupported_numeric_config_values(nSymm)
        or _contains_unsupported_numeric_config_values(count_array)
    ):
        raise ValueError("nSymm must be a finite positive integer")
    scalar = count_array.item()
    try:
        count = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("nSymm must be a finite positive integer") from exc
    if not np.isfinite(count) or not count.is_integer() or count <= 0:
        raise ValueError("nSymm must be a finite positive integer")
    return int(count)


def _validate_symmetry_offsets(symmetryOffsets: Any) -> list[float]:
    if _contains_unsupported_numeric_config_values(symmetryOffsets):
        raise ValueError("symmetryOffsets must contain only finite real numeric values")
    try:
        offsets = np.asarray(to_numpy(symmetryOffsets), dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "symmetryOffsets must contain only finite real numeric values"
        ) from exc
    if not np.all(np.isfinite(offsets)):
        raise ValueError("symmetryOffsets must contain only finite real numeric values")
    return [float(offset) for offset in offsets]


def _symmetry_offsets(nSymm, symmetryOffsets):
    if symmetryOffsets is not None:
        return _validate_symmetry_offsets(symmetryOffsets)
    if nSymm is None:
        return []
    count = _validate_symmetry_count(nSymm)
    return [2.0 * pi * index / count for index in range(count)]


def _symmetric_distance_function(
    manifold_name, additional_params, nSymm, symmetryOffsets
):
    base_name = _without_symmetry_suffix(manifold_name)
    base_distance = get_distance_function(base_name, additional_params)
    offsets = _symmetry_offsets(nSymm, symmetryOffsets)
    if not offsets:
        offsets = [0.0]

    def distance_function(xest, xtrue):
        return min(base_distance(xest, xtrue + offset) for offset in offsets)

    return distance_function


def _as_target_matrix(value, name: str) -> np.ndarray:
    value = _as_real_numeric_array(value, name)
    if value.size == 0:
        if value.ndim == 2:
            return value
        return value.reshape(0, 0)
    if value.ndim == 1:
        return value.reshape(1, -1)
    # Common MTT layouts are either (num_targets, dim) or (dim, num_targets).
    # Prefer rows as targets when the orientation is ambiguous; only transpose
    # dim-first layouts when the trailing axis is too large to be a common
    # Euclidean target dimension.
    if value.shape[0] <= 4 < value.shape[1]:
        return value.T
    return value


def _validate_mtt_cutoff_distance(value: Any) -> float:
    value_array = np.asarray(to_numpy(value))
    if (
        value_array.shape != ()
        or np.issubdtype(value_array.dtype, np.bool_)
        or _contains_unsupported_numeric_config_values(value)
        or _contains_unsupported_numeric_config_values(value_array)
    ):
        raise ValueError("cutoff_distance must be a finite nonnegative scalar")
    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError("cutoff_distance must be a finite nonnegative scalar")
    try:
        cutoff_distance = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cutoff_distance must be a finite nonnegative scalar") from exc
    if not np.isfinite(cutoff_distance) or cutoff_distance < 0.0:
        raise ValueError("cutoff_distance must be a finite nonnegative scalar")
    return cutoff_distance


def _euclidean_mtt_distance(x1, x2, *, cutoff_distance: float) -> float:
    raw_first = _as_real_numeric_array(x1, "x1")
    raw_second = _as_real_numeric_array(x2, "x2")
    if raw_first.size == 0 and raw_first.ndim == 2 and raw_second.ndim == 2:
        if raw_first.shape[1] != raw_second.shape[1]:
            raise ValueError("MTT state sets must use the same target dimension")
        return float(cutoff_distance * raw_second.shape[0])
    if raw_second.size == 0 and raw_second.ndim == 2 and raw_first.ndim == 2:
        if raw_first.shape[1] != raw_second.shape[1]:
            raise ValueError("MTT state sets must use the same target dimension")
        return float(cutoff_distance * raw_first.shape[0])
    first = _as_target_matrix(raw_first, "x1")
    second = _as_target_matrix(raw_second, "x2")
    if first.shape[0] == 0 or second.shape[0] == 0:
        return float(cutoff_distance * abs(first.shape[0] - second.shape[0]))
    if first.shape[1] != second.shape[1]:
        raise ValueError("MTT state sets must use the same target dimension")

    deltas = first[:, None, :] - second[None, :, :]
    costs = np.linalg.norm(deltas, axis=2)
    costs = np.minimum(costs, float(cutoff_distance))
    row_indices, column_indices = linear_sum_assignment(costs)
    matched_cost = float(costs[row_indices, column_indices].sum())
    missed_count = abs(first.shape[0] - second.shape[0])
    return matched_cost + float(cutoff_distance) * missed_count


def _state_component(value, index: int):
    value = asarray(value)
    if value.ndim == 1:
        return value[index]
    return value[index, :]


def _state_slice(value, start: int, stop: int):
    value = asarray(value)
    if value.ndim == 1:
        return value[start:stop]
    return value[start:stop, :]


def _angular_distance_from_inner_product(inner_product):
    return arccos(clip(inner_product, -1.0, 1.0))


def get_distance_function(
    manifold_name, additional_params=None, nSymm=None, symmetryOffsets=None
):
    normalized_name = _normalize_registry_name(manifold_name)
    registered_factory = _DISTANCE_FUNCTION_FACTORIES.get(normalized_name)
    if registered_factory is not None:
        return registered_factory(manifold_name, additional_params)

    if nSymm is not None or symmetryOffsets is not None:
        return _symmetric_distance_function(
            manifold_name, additional_params, nSymm, symmetryOffsets
        )

    if "circle" in normalized_name or "hypertorus" in normalized_name:

        def distance_function(xest, xtrue):
            return linalg.norm(
                AbstractHypertoroidalDistribution.angular_error(xest, xtrue)
            )

    elif _is_hypersphere_symmetric_name(normalized_name):

        def distance_function(x1, x2):
            return min(
                _angular_distance_from_inner_product(dot(x1, x2)),
                _angular_distance_from_inner_product(dot(x1, -x2)),
            )

    elif "hypersphere" in normalized_name:

        def distance_function(x1, x2):
            return _angular_distance_from_inner_product(dot(x1, x2))

    elif "se2bounded" in normalized_name:

        def distance_function(xest, xtrue):
            return linalg.norm(
                AbstractHypertoroidalDistribution.angular_error(
                    _state_component(xest, 0),
                    _state_component(xtrue, 0),
                )
            )

    elif "se2" in normalized_name or "se2linear" in normalized_name:

        def distance_function(x1, x2):
            return linalg.norm(_state_slice(x1, 1, 3) - _state_slice(x2, 1, 3))

    elif "se3bounded" in normalized_name:

        def distance_function(x1, x2):
            return min(
                _angular_distance_from_inner_product(dot(x1[:4], x2[:4])),
                _angular_distance_from_inner_product(dot(x1[:4], -x2[:4])),
            )

    elif "se3" in normalized_name or "se3linear" in normalized_name:

        def distance_function(x1, x2):
            return linalg.norm(_state_slice(x1, 4, 7) - _state_slice(x2, 4, 7))

    elif "euclidean" in normalized_name and "mtt" not in normalized_name:

        def distance_function(x1, x2):
            return linalg.norm(x1 - x2)

    elif "euclidean" in normalized_name and "mtt" in normalized_name:
        params = additional_params or {}
        cutoff_distance = _validate_mtt_cutoff_distance(
            params.get("cutoff_distance", 1000000.0)
        )

        def distance_function(x1, x2):
            return _euclidean_mtt_distance(
                x1,
                x2,
                cutoff_distance=cutoff_distance,
            )

    else:
        raise ValueError("Mode not recognized")

    return distance_function


__all__ = [
    "available_distance_functions",
    "get_distance_function",
    "register_distance_function",
]
