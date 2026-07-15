from __future__ import annotations

from collections.abc import Callable, Mapping
from math import pi
from typing import Any

import numpy
from pyrecest.backend import arccos, asarray, clip, dot, linalg, to_numpy
from pyrecest.distributions import AbstractHypertoroidalDistribution
from scipy.optimize import linear_sum_assignment

DistanceFactory = Callable[[str, dict[str, Any] | None], Callable[[Any, Any], float]]
_DISTANCE_FUNCTION_FACTORIES: dict[str, DistanceFactory] = {}
_UNSUPPORTED_NUMERIC_CONFIG_TYPES = (
    bool,
    numpy.bool_,
    str,
    bytes,
    bytearray,
    numpy.str_,
    numpy.bytes_,
    complex,
    numpy.complexfloating,
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
    tokens = tuple(
        token for token in normalized_name.replace("-", "_").split("_") if token
    )
    if "hypersphere" in tokens and any(
        token in {"symmetric", "symm"} for token in tokens
    ):
        return True

    compact = "".join(tokens)
    return (
        "hyperspheresymmetric" in compact
        or "hyperspheresymm" in compact
        or compact in {"symmetrichypersphere", "symmhypersphere"}
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
        raw_values = numpy.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        raw_values = ()
    if any(isinstance(item, _UNSUPPORTED_NUMERIC_CONFIG_TYPES) for item in raw_values):
        return True
    try:
        values = numpy.asarray(to_numpy(value), dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _UNSUPPORTED_NUMERIC_CONFIG_TYPES) for item in values)


def _as_real_numeric_array(value: Any, name: str) -> numpy.ndarray:
    message = f"{name} must contain only finite real numeric values"
    if _contains_unsupported_numeric_config_values(value):
        raise ValueError(message)
    try:
        values = numpy.asarray(to_numpy(value), dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not numpy.all(numpy.isfinite(values)):
        raise ValueError(message)
    return values


def _validate_symmetry_count(nSymm: Any) -> int:
    count_array = numpy.asarray(to_numpy(nSymm))
    if (
        count_array.shape != ()
        or numpy.issubdtype(count_array.dtype, numpy.bool_)
        or _contains_unsupported_numeric_config_values(nSymm)
        or _contains_unsupported_numeric_config_values(count_array)
    ):
        raise ValueError("nSymm must be a finite positive integer")
    scalar = count_array.item()
    try:
        count = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("nSymm must be a finite positive integer") from exc
    if not numpy.isfinite(count) or not count.is_integer() or count <= 0:
        raise ValueError("nSymm must be a finite positive integer")
    return int(count)


def _validate_symmetry_offsets(symmetryOffsets: Any) -> list[float]:
    if _contains_unsupported_numeric_config_values(symmetryOffsets):
        raise ValueError("symmetryOffsets must contain only finite real numeric values")
    try:
        offsets = numpy.asarray(to_numpy(symmetryOffsets), dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "symmetryOffsets must contain only finite real numeric values"
        ) from exc
    if not numpy.all(numpy.isfinite(offsets)):
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
        xest_array = asarray(xest)
        xtrue_array = asarray(xtrue)
        return min(
            base_distance(xest_array, xtrue_array + offset) for offset in offsets
        )

    return distance_function


def _as_target_matrix(
    value, name: str, *, expected_target_dim: int | None = None
) -> numpy.ndarray:
    value = _as_real_numeric_array(value, name)
    if value.ndim not in (1, 2):
        raise ValueError(f"{name} must be a one- or two-dimensional target set")
    if value.size == 0:
        if value.ndim == 2:
            return value
        return value.reshape(0, 0)
    if value.ndim == 1:
        return value.reshape(1, -1)
    if expected_target_dim is not None:
        if value.shape[1] == expected_target_dim:
            return value
        if value.shape[0] == expected_target_dim:
            return value.T
    # Common MTT layouts are either (num_targets, dim) or (dim, num_targets).
    # Prefer rows as targets when the orientation is ambiguous; only transpose
    # dim-first layouts when the trailing axis is too large to be a common
    # Euclidean target dimension.
    if value.shape[0] <= 4 < value.shape[1]:
        return value.T
    return value


def _validate_mtt_cutoff_distance(value: Any) -> float:
    value_array = numpy.asarray(to_numpy(value))
    if (
        value_array.shape != ()
        or numpy.issubdtype(value_array.dtype, numpy.bool_)
        or _contains_unsupported_numeric_config_values(value)
        or _contains_unsupported_numeric_config_values(value_array)
    ):
        raise ValueError("cutoff_distance must be a finite nonnegative scalar")
    scalar = value_array.item()
    if isinstance(scalar, (bool, numpy.bool_)):
        raise ValueError("cutoff_distance must be a finite nonnegative scalar")
    try:
        cutoff_distance = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cutoff_distance must be a finite nonnegative scalar") from exc
    if not numpy.isfinite(cutoff_distance) or cutoff_distance < 0.0:
        raise ValueError("cutoff_distance must be a finite nonnegative scalar")
    return cutoff_distance


def _coerce_additional_params(additional_params: Any) -> Mapping[str, Any]:
    if additional_params is None:
        return {}
    if not isinstance(additional_params, Mapping):
        raise ValueError("additional_params must be a mapping or None")
    return additional_params


def _euclidean_mtt_distance(x1, x2, *, cutoff_distance: float) -> float:
    first = _as_target_matrix(x1, "x1")
    second = _as_target_matrix(x2, "x2")
    if first.shape[0] == 0 and first.shape[1] != 0:
        second = _as_target_matrix(x2, "x2", expected_target_dim=first.shape[1])
    elif second.shape[0] == 0 and second.shape[1] != 0:
        first = _as_target_matrix(x1, "x1", expected_target_dim=second.shape[1])
    if first.shape[0] == 0 or second.shape[0] == 0:
        if (
            first.shape[1] != 0
            and second.shape[1] != 0
            and first.shape[1] != second.shape[1]
        ):
            raise ValueError("MTT state sets must use the same target dimension")
        return float(cutoff_distance * abs(first.shape[0] - second.shape[0]))
    if first.shape[1] != second.shape[1]:
        raise ValueError("MTT state sets must use the same target dimension")

    deltas = first[:, None, :] - second[None, :, :]
    costs = numpy.linalg.norm(deltas, axis=2)
    costs = numpy.minimum(costs, float(cutoff_distance))
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
            x1_array = asarray(x1)
            x2_array = asarray(x2)
            return min(
                _angular_distance_from_inner_product(dot(x1_array, x2_array)),
                _angular_distance_from_inner_product(dot(x1_array, -x2_array)),
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
            orientation1 = _state_slice(x1, 0, 4)
            orientation2 = _state_slice(x2, 0, 4)
            return min(
                _angular_distance_from_inner_product(dot(orientation1, orientation2)),
                _angular_distance_from_inner_product(dot(orientation1, -orientation2)),
            )

    elif "se3" in normalized_name or "se3linear" in normalized_name:

        def distance_function(x1, x2):
            return linalg.norm(_state_slice(x1, 4, 7) - _state_slice(x2, 4, 7))

    elif "euclidean" in normalized_name and "mtt" not in normalized_name:

        def distance_function(x1, x2):
            return linalg.norm(asarray(x1) - asarray(x2))

    elif "euclidean" in normalized_name and "mtt" in normalized_name:
        params = _coerce_additional_params(additional_params)
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
