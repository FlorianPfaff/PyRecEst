"""Deterministic point-set metrics for sampled shapes and extended objects.

The functions in this module are intentionally NumPy/SciPy based. They are
designed for evaluation and diagnostics, not differentiable model code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_DEFAULT_CHUNK_SIZE = 4096


def as_point_set(
    points: Any, *, name: str = "points", expected_dim: int | None = None
) -> np.ndarray:
    """Return ``points`` as a finite ``float64`` array with shape ``(N, D)``.

    Parameters
    ----------
    points:
        Array-like point set.
    name:
        Name used in validation error messages.
    expected_dim:
        Optional required point dimensionality.
    """

    try:
        array = np.asarray(points)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (N, D).") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (N, D).")
    if expected_dim is not None and array.shape[1] != expected_dim:
        raise ValueError(f"{name} must have shape (N, {expected_dim}).")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point.")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain only finite real numeric values.")
    try:
        array = array.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must contain only finite real numeric values."
        ) from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite real numeric values.")
    return array


def deterministic_subsample(
    points: Any, *, max_points: int | None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic random subset of ``points`` and its indices.

    If ``max_points`` is ``None``, non-positive, or at least the number of
    points, the original order is preserved and all indices are returned.
    Otherwise, ``max_points`` unique indices are sampled without replacement and
    sorted so downstream computations are reproducible and stable under
    independent chunk sizes.
    """

    point_array = as_point_set(points)
    max_points_int = _normalize_optional_integer(max_points, "max_points")
    if (
        max_points_int is None
        or max_points_int <= 0
        or point_array.shape[0] <= max_points_int
    ):
        indices = np.arange(point_array.shape[0], dtype=np.int64)
        return point_array, indices

    rng = np.random.default_rng(seed)
    indices = np.sort(
        rng.choice(point_array.shape[0], size=max_points_int, replace=False)
    ).astype(np.int64)
    return point_array[indices], indices


def nearest_neighbor_distances(
    query: Any,
    reference: Any,
    *,
    query_chunk_size: int = _DEFAULT_CHUNK_SIZE,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Distance from each query point to its nearest reference point.

    The function uses :class:`scipy.spatial.cKDTree` when available and falls
    back to a deterministic chunked dense implementation otherwise.
    """

    query_array = as_point_set(query, name="query")
    reference_array = as_point_set(reference, name="reference")
    _validate_matching_dimension(query_array, reference_array)
    query_chunk_size = _validate_chunk_size(query_chunk_size)
    return_indices = _validate_boolean_flag(return_indices, "return_indices")

    tree_result = _nearest_neighbor_distances_ckdtree(
        query_array,
        reference_array,
        query_chunk_size=query_chunk_size,
        return_indices=return_indices,
    )
    if tree_result is not None:
        return tree_result
    return _nearest_neighbor_distances_dense(
        query_array,
        reference_array,
        query_chunk_size=query_chunk_size,
        return_indices=return_indices,
    )


def chamfer_distance(
    points_a: Any,
    points_b: Any,
    *,
    squared: bool = False,
    symmetric: bool = True,
    query_chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> float:
    """Return directed or symmetric Chamfer distance between two point sets.

    ``symmetric=True`` returns the sum of the two directed mean nearest-neighbor
    distances, matching the convention used by common 3D reconstruction
    diagnostics. ``squared=True`` applies the same convention to squared
    nearest-neighbor distances.
    """

    squared = _validate_boolean_flag(squared, "squared")
    symmetric = _validate_boolean_flag(symmetric, "symmetric")

    a_to_b = nearest_neighbor_distances(
        points_a, points_b, query_chunk_size=query_chunk_size
    )
    value = float(np.mean(a_to_b**2 if squared else a_to_b))
    if not symmetric:
        return value
    b_to_a = nearest_neighbor_distances(
        points_b, points_a, query_chunk_size=query_chunk_size
    )
    return value + float(np.mean(b_to_a**2 if squared else b_to_a))


def precision_recall_fscore(
    estimate_points: Any,
    reference_points: Any,
    threshold: float,
    *,
    query_chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> dict[str, float]:
    """Return precision, recall, and F-score for a distance threshold."""

    threshold_float = _validate_threshold(threshold)
    estimate_to_reference = nearest_neighbor_distances(
        estimate_points, reference_points, query_chunk_size=query_chunk_size
    )
    reference_to_estimate = nearest_neighbor_distances(
        reference_points, estimate_points, query_chunk_size=query_chunk_size
    )
    precision = float(np.mean(estimate_to_reference <= threshold_float))
    recall = float(np.mean(reference_to_estimate <= threshold_float))
    f_score = (
        0.0
        if precision + recall <= 0.0
        else float(2.0 * precision * recall / (precision + recall))
    )
    return {
        "threshold": threshold_float,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
    }


def precision_recall_curve(
    estimate_points: Any,
    reference_points: Any,
    thresholds: Sequence[float],
    *,
    query_chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> list[dict[str, float]]:
    """Return precision/recall/F-score rows for multiple thresholds."""

    threshold_values = tuple(_validate_threshold(threshold) for threshold in thresholds)
    if not threshold_values:
        raise ValueError("At least one threshold is required.")
    estimate_to_reference = nearest_neighbor_distances(
        estimate_points, reference_points, query_chunk_size=query_chunk_size
    )
    reference_to_estimate = nearest_neighbor_distances(
        reference_points, estimate_points, query_chunk_size=query_chunk_size
    )
    rows = []
    for threshold in threshold_values:
        precision = float(np.mean(estimate_to_reference <= threshold))
        recall = float(np.mean(reference_to_estimate <= threshold))
        f_score = (
            0.0
            if precision + recall <= 0.0
            else float(2.0 * precision * recall / (precision + recall))
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
            }
        )
    return rows


def point_set_geometry_summary(
    estimate_points: Any,
    reference_points: Any,
    *,
    thresholds: Sequence[float] = (0.1,),
    query_chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Return standard directed/symmetric point-set geometry metrics.

    The summary keys follow reconstruction terminology:

    - ``accuracy_*``: estimate-to-reference nearest-neighbor distances.
    - ``completion_*``: reference-to-estimate nearest-neighbor distances.
    - ``chamfer_l1``: sum of directed mean distances.
    - ``chamfer_l2``: sum of directed mean squared distances.
    """

    threshold_values = tuple(_validate_threshold(threshold) for threshold in thresholds)
    if not threshold_values:
        raise ValueError("At least one threshold is required.")

    estimate_to_reference = nearest_neighbor_distances(
        estimate_points, reference_points, query_chunk_size=query_chunk_size
    )
    reference_to_estimate = nearest_neighbor_distances(
        reference_points, estimate_points, query_chunk_size=query_chunk_size
    )

    summary = {
        "accuracy_mean": float(estimate_to_reference.mean()),
        "accuracy_median": float(np.median(estimate_to_reference)),
        "accuracy_rmse": float(np.sqrt(np.mean(estimate_to_reference**2))),
        "completion_mean": float(reference_to_estimate.mean()),
        "completion_median": float(np.median(reference_to_estimate)),
        "completion_rmse": float(np.sqrt(np.mean(reference_to_estimate**2))),
        "chamfer_l1": float(
            estimate_to_reference.mean() + reference_to_estimate.mean()
        ),
        "chamfer_l2": float(
            np.mean(estimate_to_reference**2) + np.mean(reference_to_estimate**2)
        ),
    }

    threshold_rows = []
    for threshold in threshold_values:
        precision = float(np.mean(estimate_to_reference <= threshold))
        recall = float(np.mean(reference_to_estimate <= threshold))
        f_score = (
            0.0
            if precision + recall <= 0.0
            else float(2.0 * precision * recall / (precision + recall))
        )
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
            }
        )
    return summary, threshold_rows


def distance_quantiles(
    query: Any,
    reference: Any,
    *,
    quantiles: Sequence[float] = (0.5, 0.9, 0.95, 0.99),
    query_chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> dict[float, float]:
    """Return quantiles of directed nearest-neighbor distances."""

    quantile_values = tuple(_validate_quantile(quantile) for quantile in quantiles)
    if not quantile_values:
        raise ValueError("At least one quantile is required.")
    distances = nearest_neighbor_distances(
        query, reference, query_chunk_size=query_chunk_size
    )
    values = np.quantile(distances, quantile_values)
    return {
        quantile: float(value)
        for quantile, value in zip(quantile_values, np.atleast_1d(values), strict=True)
    }


def _nearest_neighbor_distances_ckdtree(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    query_chunk_size: int,
    return_indices: bool,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]
    except ImportError:
        return None

    tree = cKDTree(reference)
    distances = np.empty((query.shape[0],), dtype=np.float64)
    indices = np.empty((query.shape[0],), dtype=np.int64) if return_indices else None
    for start in range(0, query.shape[0], query_chunk_size):
        chunk = query[start : start + query_chunk_size]
        chunk_distances, chunk_indices = tree.query(chunk, k=1)
        distances[start : start + chunk.shape[0]] = chunk_distances
        if indices is not None:
            indices[start : start + chunk.shape[0]] = chunk_indices
    if indices is None:
        return distances
    return distances, indices


def _nearest_neighbor_distances_dense(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    query_chunk_size: int,
    return_indices: bool,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    ref_t = reference.T
    ref_norm = np.sum(reference**2, axis=1)
    distances = np.empty((query.shape[0],), dtype=np.float64)
    indices = np.empty((query.shape[0],), dtype=np.int64) if return_indices else None
    for start in range(0, query.shape[0], query_chunk_size):
        chunk = query[start : start + query_chunk_size]
        chunk_norm = np.sum(chunk**2, axis=1, keepdims=True)
        squared = np.maximum(
            chunk_norm + ref_norm[None, :] - 2.0 * (chunk @ ref_t), 0.0
        )
        chunk_indices = np.argmin(squared, axis=1)
        distances[start : start + chunk.shape[0]] = np.sqrt(
            squared[np.arange(chunk.shape[0]), chunk_indices]
        )
        if indices is not None:
            indices[start : start + chunk.shape[0]] = chunk_indices
    if indices is None:
        return distances
    return distances, indices


def _validate_matching_dimension(query: np.ndarray, reference: np.ndarray) -> None:
    if query.shape[1] != reference.shape[1]:
        raise ValueError(
            f"query and reference must have the same point dimension, got {query.shape[1]} and {reference.shape[1]}."
        )


def _validate_boolean_flag(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    value_array = np.asarray(value)
    if value_array.shape == () and value_array.dtype == np.bool_:
        return bool(value_array.item())
    raise TypeError(f"{name} must be a boolean.")


def _is_temporal_scalar_array(array: np.ndarray) -> bool:
    if array.shape != ():
        return False
    if array.dtype.kind in "Mm":
        return True
    if array.dtype != object:
        return False
    try:
        scalar = array.item()
    except (TypeError, ValueError):
        return False
    return isinstance(scalar, (np.datetime64, np.timedelta64))


def _validate_chunk_size(query_chunk_size: int) -> int:
    array = np.asarray(query_chunk_size)
    if array.shape != () or array.dtype == np.bool_ or _is_temporal_scalar_array(array):
        raise ValueError("query_chunk_size must be a positive integer.")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_, str, bytes)):
        raise ValueError("query_chunk_size must be a positive integer.")
    try:
        chunk_size_float = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("query_chunk_size must be a positive integer.") from exc
    if (
        not np.isfinite(chunk_size_float)
        or not chunk_size_float.is_integer()
        or chunk_size_float < 1.0
    ):
        raise ValueError("query_chunk_size must be a positive integer.")
    return int(chunk_size_float)


def _normalize_optional_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None

    value_array = np.asarray(value)
    message = f"{name} must be None or an integer."
    if (
        value_array.shape != ()
        or value_array.dtype == np.bool_
        or _is_temporal_scalar_array(value_array)
    ):
        raise ValueError(message)

    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_, str, bytes)):
        raise ValueError(message)
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)

    try:
        scalar_float = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(scalar_float) or not scalar_float.is_integer():
        raise ValueError(message)
    return int(scalar_float)


def _validate_numeric_scalar(value: Any, message: str) -> float:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.shape != () or array.dtype == np.bool_ or _is_temporal_scalar_array(array):
        raise ValueError(message)

    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_, str, bytes)):
        raise ValueError(message)
    try:
        value_float = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(value_float):
        raise ValueError(message)
    return value_float


def _validate_threshold(threshold: float) -> float:
    message = "Distance thresholds must be finite and non-negative numeric scalars."
    threshold_float = _validate_numeric_scalar(threshold, message)
    if threshold_float < 0.0:
        raise ValueError(message)
    return threshold_float


def _validate_quantile(quantile: float) -> float:
    message = "Quantiles must be finite numeric scalars in [0, 1]."
    quantile_float = _validate_numeric_scalar(quantile, message)
    if quantile_float < 0.0 or quantile_float > 1.0:
        raise ValueError("Quantiles must be in [0, 1].")
    return quantile_float
