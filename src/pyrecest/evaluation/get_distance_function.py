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


def register_distance_function(
    manifold_name: str, factory: DistanceFactory
) -> DistanceFactory:
    """Register a custom distance-function factory for a manifold name."""
    if not manifold_name:
        raise ValueError("manifold_name must be a non-empty string")
    _DISTANCE_FUNCTION_FACTORIES[manifold_name.lower()] = factory
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


def _symmetry_offsets(nSymm, symmetryOffsets):
    if symmetryOffsets is not None:
        return list(asarray(symmetryOffsets))
    if nSymm is None:
        return []
    return [2.0 * pi * index / int(nSymm) for index in range(int(nSymm))]


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


def _as_target_matrix(value) -> np.ndarray:
    value = np.asarray(to_numpy(value), dtype=float)
    if value.size == 0:
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


def _euclidean_mtt_distance(x1, x2, *, cutoff_distance: float) -> float:
    first = _as_target_matrix(x1)
    second = _as_target_matrix(x2)
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


def _angular_distance_from_inner_product(inner_product):
    return arccos(clip(inner_product, -1.0, 1.0))


def get_distance_function(
    manifold_name, additional_params=None, nSymm=None, symmetryOffsets=None
):
    normalized_name = str(manifold_name).lower()
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
                AbstractHypertoroidalDistribution.angular_error(xest[0, :], xtrue[0, :])
            )

    elif "se2" in normalized_name or "se2linear" in normalized_name:

        def distance_function(x1, x2):
            return linalg.norm(x1[1:3, :] - x2[1:3, :])

    elif "se3bounded" in normalized_name:

        def distance_function(x1, x2):
            return min(
                _angular_distance_from_inner_product(dot(x1[:4], x2[:4])),
                _angular_distance_from_inner_product(dot(x1[:4], -x2[:4])),
            )

    elif "se3" in normalized_name or "se3linear" in normalized_name:

        def distance_function(x1, x2):
            return linalg.norm(x1[4:7, :] - x2[4:7, :])

    elif "euclidean" in normalized_name and "mtt" not in normalized_name:

        def distance_function(x1, x2):
            return linalg.norm(x1 - x2)

    elif "euclidean" in normalized_name and "mtt" in normalized_name:
        params = additional_params or {}
        cutoff_distance = float(params.get("cutoff_distance", 1000000.0))

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
