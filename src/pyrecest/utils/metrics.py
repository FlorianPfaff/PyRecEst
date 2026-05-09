"""Estimator, finite-set, and extended-object performance metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from shapely import Polygon

ArrayLike = Any
DistanceFunction = Callable[[np.ndarray, np.ndarray], float]

__all__ = [
    "anis",
    "anees",
    "average_nees",
    "average_nis",
    "chi_square_confidence_bounds",
    "chi_square_confidence_interval",
    "consistency_fraction",
    "eot_shape_iou",
    "error_vectors",
    "extent_error",
    "extent_intersection_over_union",
    "extent_matrix_error",
    "extent_wasserstein_distance",
    "gaussian_wasserstein_distance",
    "gospa_distance",
    "iou_polygon",
    "is_chi_square_consistent",
    "is_within_chi_square_confidence_interval",
    "mae",
    "mean_absolute_error",
    "mean_squared_error",
    "mospa_distance",
    "mse",
    "nees",
    "nees_confidence_bounds",
    "nees_confidence_interval",
    "nis",
    "nis_confidence_bounds",
    "nis_confidence_interval",
    "normalized_estimation_error_squared",
    "normalized_innovation_squared",
    "ospa_distance",
    "rmse",
    "root_mean_squared_error",
    "squared_error",
]


def error_vectors(estimates: ArrayLike, groundtruths: ArrayLike) -> np.ndarray:
    """Return estimate-minus-truth error vectors."""
    estimates_array = _as_numeric_array(estimates, "estimates")
    groundtruths_array = _as_numeric_array(groundtruths, "groundtruths")
    if estimates_array.shape != groundtruths_array.shape:
        raise ValueError("estimates and groundtruths must have identical shapes")
    return estimates_array - groundtruths_array


def squared_error(estimates: ArrayLike, groundtruths: ArrayLike) -> np.ndarray:
    """Return per-sample squared Euclidean errors."""
    errors = _as_sample_matrix(error_vectors(estimates, groundtruths), "errors")
    return np.sum(errors * errors, axis=-1)


def mean_squared_error(estimates: ArrayLike, groundtruths: ArrayLike, *, axis: int | tuple[int, ...] | None = None):
    """Return mean squared component error.

    ``axis=None`` averages all entries. Use ``axis=0`` to obtain a component-wise
    MSE over samples.
    """
    errors = error_vectors(estimates, groundtruths)
    return np.mean(errors * errors, axis=axis)


def root_mean_squared_error(estimates: ArrayLike, groundtruths: ArrayLike, *, axis: int | tuple[int, ...] | None = None):
    """Return root mean squared component error."""
    return np.sqrt(mean_squared_error(estimates, groundtruths, axis=axis))


def mean_absolute_error(estimates: ArrayLike, groundtruths: ArrayLike, *, axis: int | tuple[int, ...] | None = None):
    """Return mean absolute component error."""
    return np.mean(np.abs(error_vectors(estimates, groundtruths)), axis=axis)


mse = mean_squared_error
rmse = root_mean_squared_error
mae = mean_absolute_error


def normalized_estimation_error_squared(estimates: ArrayLike, uncertainties: ArrayLike, groundtruths: ArrayLike | None = None):
    """Return normalized estimation error squared values.

    If ``groundtruths`` is omitted, ``estimates`` is interpreted as an error
    vector or matrix. Otherwise the error is ``estimates - groundtruths``. The
    returned values have expected value equal to the state dimension for a
    consistent covariance.
    """
    errors = _as_numeric_array(estimates, "estimates") if groundtruths is None else error_vectors(estimates, groundtruths)
    errors = _as_sample_matrix(errors, "errors")
    covariances = _as_covariance_stack(uncertainties, errors.shape[0], errors.shape[1], "uncertainties")
    values = _quadratic_forms(errors, covariances)
    return float(values[0]) if _is_single_vector(estimates) and groundtruths is None else values


def average_nees(estimates: ArrayLike, uncertainties: ArrayLike, groundtruths: ArrayLike | None = None) -> float:
    """Return average normalized estimation error squared."""
    return float(np.mean(np.atleast_1d(normalized_estimation_error_squared(estimates, uncertainties, groundtruths))))


def anees(estimates: ArrayLike, uncertainties: ArrayLike, groundtruths: ArrayLike) -> float:
    """Backward-compatible alias for average NEES."""
    return average_nees(estimates, uncertainties, groundtruths)


nees = normalized_estimation_error_squared


def normalized_innovation_squared(innovations_or_measurements: ArrayLike, predicted_or_covariances: ArrayLike, innovation_covariances: ArrayLike | None = None):
    """Return normalized innovation squared values.

    Two call styles are supported:

    - ``nis(innovations, innovation_covariances)``
    - ``nis(measurements, predicted_measurements, innovation_covariances)``
    """
    if innovation_covariances is None:
        innovations = _as_numeric_array(innovations_or_measurements, "innovations")
        covariances = predicted_or_covariances
    else:
        innovations = error_vectors(innovations_or_measurements, predicted_or_covariances)
        covariances = innovation_covariances
    innovations = _as_sample_matrix(innovations, "innovations")
    covariance_stack = _as_covariance_stack(covariances, innovations.shape[0], innovations.shape[1], "innovation_covariances")
    values = _quadratic_forms(innovations, covariance_stack)
    return float(values[0]) if innovations.shape[0] == 1 and _is_single_vector(innovations_or_measurements) else values


def average_nis(innovations_or_measurements: ArrayLike, predicted_or_covariances: ArrayLike, innovation_covariances: ArrayLike | None = None) -> float:
    """Return average normalized innovation squared."""
    return float(np.mean(np.atleast_1d(normalized_innovation_squared(innovations_or_measurements, predicted_or_covariances, innovation_covariances))))


nis = normalized_innovation_squared
anis = average_nis


def chi_square_confidence_bounds(degrees_of_freedom: int, *, n_samples: int = 1, confidence: float = 0.95) -> tuple[float, float]:
    """Return two-sided chi-square bounds for an averaged NEES/NIS statistic."""
    if int(degrees_of_freedom) <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if int(n_samples) <= 0:
        raise ValueError("n_samples must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in the open interval (0, 1)")
    alpha = 1.0 - float(confidence)
    aggregate_dof = int(degrees_of_freedom) * int(n_samples)
    lower = chi2.ppf(alpha / 2.0, aggregate_dof) / float(n_samples)
    upper = chi2.ppf(1.0 - alpha / 2.0, aggregate_dof) / float(n_samples)
    return float(lower), float(upper)


def chi_square_confidence_interval(degrees_of_freedom: int, *, n_samples: int = 1, confidence: float = 0.95) -> tuple[float, float]:
    """Alias for :func:`chi_square_confidence_bounds`."""
    return chi_square_confidence_bounds(degrees_of_freedom, n_samples=n_samples, confidence=confidence)


def nees_confidence_bounds(state_dim: int, *, n_samples: int = 1, confidence: float = 0.95) -> tuple[float, float]:
    """Return chi-square consistency bounds for NEES or ANEES."""
    return chi_square_confidence_bounds(state_dim, n_samples=n_samples, confidence=confidence)


def nees_confidence_interval(state_dim: int, *, n_samples: int = 1, confidence: float = 0.95) -> tuple[float, float]:
    """Alias for :func:`nees_confidence_bounds`."""
    return nees_confidence_bounds(state_dim, n_samples=n_samples, confidence=confidence)


def nis_confidence_bounds(measurement_dim: int, *, n_samples: int = 1, confidence: float = 0.95) -> tuple[float, float]:
    """Return chi-square consistency bounds for NIS or ANIS."""
    return chi_square_confidence_bounds(measurement_dim, n_samples=n_samples, confidence=confidence)


def nis_confidence_interval(measurement_dim: int, *, n_samples: int = 1, confidence: float = 0.95) -> tuple[float, float]:
    """Alias for :func:`nis_confidence_bounds`."""
    return nis_confidence_bounds(measurement_dim, n_samples=n_samples, confidence=confidence)


def is_chi_square_consistent(statistic: float, degrees_of_freedom: int, *, n_samples: int = 1, confidence: float = 0.95) -> bool:
    """Return whether a scalar statistic lies inside chi-square bounds."""
    lower, upper = chi_square_confidence_bounds(degrees_of_freedom, n_samples=n_samples, confidence=confidence)
    return bool(lower <= float(statistic) <= upper)


def is_within_chi_square_confidence_interval(statistic: float, degrees_of_freedom: int, *, n_samples: int = 1, confidence: float = 0.95) -> bool:
    """Alias for :func:`is_chi_square_consistent`."""
    return is_chi_square_consistent(statistic, degrees_of_freedom, n_samples=n_samples, confidence=confidence)


def consistency_fraction(values: ArrayLike, lower: float, upper: float) -> float:
    """Return the fraction of scalar values inside ``[lower, upper]``."""
    values_array = _as_numeric_array(values, "values")
    if values_array.size == 0:
        return 0.0
    return float(np.mean((values_array >= float(lower)) & (values_array <= float(upper))))


def ospa_distance(
    estimated_points: ArrayLike,
    reference_points: ArrayLike,
    *,
    cutoff: float,
    order: float = 1.0,
    distance_fn: DistanceFunction | None = None,
    return_components: bool = False,
) -> float | dict[str, float | int]:
    """Return the optimal sub-pattern assignment distance between two finite sets."""
    order, cutoff = _validate_order_cutoff(order, cutoff)
    estimated, reference = _coerce_point_sets(estimated_points, reference_points)
    n_estimated = estimated.shape[0]
    n_reference = reference.shape[0]
    normalizer = max(n_estimated, n_reference)
    if normalizer == 0:
        return _components_or_distance("ospa", 0.0, 0.0, 0.0, 0, return_components)
    if n_estimated == 0 or n_reference == 0:
        return _components_or_distance("ospa", cutoff, 0.0, cutoff, 0, return_components)

    assignment_cost, assignments = _clipped_assignment_cost(estimated, reference, order=order, cutoff=cutoff, distance_fn=distance_fn)
    cardinality_cost = cutoff**order * abs(n_estimated - n_reference)
    localization_component = assignment_cost / float(normalizer)
    cardinality_component = cardinality_cost / float(normalizer)
    distance = (localization_component + cardinality_component) ** (1.0 / order)
    return _components_or_distance("ospa", distance, localization_component ** (1.0 / order), cardinality_component ** (1.0 / order), len(assignments), return_components)


def mospa_distance(
    estimated_point_sets: Sequence[ArrayLike],
    reference_point_sets: Sequence[ArrayLike],
    *,
    cutoff: float,
    order: float = 1.0,
    distance_fn: DistanceFunction | None = None,
    return_per_step: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Return mean OSPA over a sequence of finite-set estimates."""
    if len(estimated_point_sets) != len(reference_point_sets):
        raise ValueError("estimated_point_sets and reference_point_sets must have the same length")
    distances = np.asarray(
        [ospa_distance(estimated, reference, cutoff=cutoff, order=order, distance_fn=distance_fn) for estimated, reference in zip(estimated_point_sets, reference_point_sets)],
        dtype=float,
    )
    mean_distance = float(np.mean(distances)) if distances.size else 0.0
    return (mean_distance, distances) if return_per_step else mean_distance


def gospa_distance(
    estimated_points: ArrayLike,
    reference_points: ArrayLike,
    *,
    cutoff: float,
    order: float = 1.0,
    alpha: float = 2.0,
    distance_fn: DistanceFunction | None = None,
    return_components: bool = False,
) -> float | dict[str, float | int]:
    """Return the generalized OSPA distance between two finite sets."""
    order, cutoff = _validate_order_cutoff(order, cutoff)
    if not 0.0 < float(alpha) <= 2.0:
        raise ValueError("alpha must lie in the interval (0, 2]")
    estimated, reference = _coerce_point_sets(estimated_points, reference_points)
    if estimated.shape[0] == 0 and reference.shape[0] == 0:
        return _components_or_distance("gospa", 0.0, 0.0, 0.0, 0, return_components)

    assignment_cost, assignments = _clipped_assignment_cost(estimated, reference, order=order, cutoff=cutoff, distance_fn=distance_fn)
    cardinality_cost = cutoff**order / float(alpha) * abs(estimated.shape[0] - reference.shape[0])
    distance = (assignment_cost + cardinality_cost) ** (1.0 / order)
    return _components_or_distance("gospa", distance, assignment_cost ** (1.0 / order), cardinality_cost ** (1.0 / order), len(assignments), return_components)


def iou_polygon(polygon1, polygon2):
    """Return intersection over union for two polygonal shapes."""
    if not isinstance(polygon1, Polygon):
        polygon1 = Polygon(polygon1)
    if not isinstance(polygon2, Polygon):
        polygon2 = Polygon(polygon2)
    intersection = polygon1.intersection(polygon2)
    union = polygon1.union(polygon2)
    return float(intersection.area / union.area) if union.area > 0 else 0.0


def eot_shape_iou(shape1, shape2) -> float:
    """Return polygon IoU for extended-object shape estimates."""
    return float(iou_polygon(shape1, shape2))


def extent_intersection_over_union(shape1, shape2) -> float:
    """Alias for polygon IoU used by extended-object tracking metrics."""
    return eot_shape_iou(shape1, shape2)


def gaussian_wasserstein_distance(mean1: ArrayLike, covariance1: ArrayLike, mean2: ArrayLike, covariance2: ArrayLike, *, squared: bool = False) -> float:
    """Return the 2-Wasserstein distance between Gaussian distributions."""
    mean1_np = _as_numeric_array(mean1, "mean1").reshape(-1)
    mean2_np = _as_numeric_array(mean2, "mean2").reshape(-1)
    covariance1_np = _symmetrize(_as_numeric_array(covariance1, "covariance1"))
    covariance2_np = _symmetrize(_as_numeric_array(covariance2, "covariance2"))
    if mean1_np.shape != mean2_np.shape:
        raise ValueError("mean1 and mean2 must have the same shape")
    _validate_square_matrix(covariance1_np, "covariance1")
    _validate_square_matrix(covariance2_np, "covariance2")
    if covariance1_np.shape != covariance2_np.shape:
        raise ValueError("covariance1 and covariance2 must have the same shape")
    if covariance1_np.shape[0] != mean1_np.size:
        raise ValueError("covariance dimensions must match mean dimensions")

    mean_error = mean1_np - mean2_np
    mean_term = float(mean_error @ mean_error)
    covariance1_sqrt = _symmetric_matrix_square_root(covariance1_np)
    middle_sqrt = _symmetric_matrix_square_root(covariance1_sqrt @ covariance2_np @ covariance1_sqrt)
    trace_term = float(np.trace(covariance1_np + covariance2_np - 2.0 * middle_sqrt))
    squared_distance = max(mean_term + trace_term, 0.0)
    return float(squared_distance) if squared else float(np.sqrt(squared_distance))


def extent_wasserstein_distance(estimated_extent: ArrayLike, reference_extent: ArrayLike, *, squared: bool = False) -> float:
    """Return the covariance/extent part of the Gaussian 2-Wasserstein distance."""
    estimated = _as_numeric_array(estimated_extent, "estimated_extent")
    reference = _as_numeric_array(reference_extent, "reference_extent")
    _validate_square_matrix(estimated, "estimated_extent")
    _validate_square_matrix(reference, "reference_extent")
    if estimated.shape != reference.shape:
        raise ValueError("estimated_extent and reference_extent must have the same shape")
    zeros = np.zeros(estimated.shape[0], dtype=float)
    return gaussian_wasserstein_distance(zeros, estimated, zeros, reference, squared=squared)


def extent_matrix_error(estimated_extent: ArrayLike, reference_extent: ArrayLike, *, ord: str | int | float = "fro", relative: bool = False) -> float:
    """Return matrix-norm error between estimated and reference extents."""
    estimated = _as_numeric_array(estimated_extent, "estimated_extent")
    reference = _as_numeric_array(reference_extent, "reference_extent")
    if estimated.shape != reference.shape:
        raise ValueError("estimated_extent and reference_extent must have the same shape")
    error = float(np.linalg.norm(estimated - reference, ord=ord))
    if not relative:
        return error
    denominator = float(np.linalg.norm(reference, ord=ord))
    if denominator == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return error / denominator


def extent_error(estimated_extent: ArrayLike, reference_extent: ArrayLike, *, ord: str | int | float = "fro", relative: bool = False) -> float:
    """Alias for :func:`extent_matrix_error`."""
    return extent_matrix_error(estimated_extent, reference_extent, ord=ord, relative=relative)


def _as_numeric_array(value: ArrayLike, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except TypeError:
            pass
    try:
        from pyrecest.backend import to_numpy  # pylint: disable=import-outside-toplevel

        value = to_numpy(value)
    except (ImportError, AttributeError, TypeError):
        pass
    array = np.asarray(value, dtype=float)
    if array.dtype == object:
        raise TypeError(f"{name} could not be converted to a numeric array")
    return array


def _as_sample_matrix(values: ArrayLike, name: str) -> np.ndarray:
    array = _as_numeric_array(values, name)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (dim,) or (n, dim)")
    return array


def _is_single_vector(values: ArrayLike) -> bool:
    return _as_numeric_array(values, "values").ndim == 1


def _as_covariance_stack(covariances: ArrayLike, n_samples: int, dim: int, name: str) -> np.ndarray:
    covariance_array = _as_numeric_array(covariances, name)
    if covariance_array.ndim == 2:
        if covariance_array.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim})")
        return np.broadcast_to(covariance_array, (n_samples, dim, dim))
    if covariance_array.ndim == 3:
        if covariance_array.shape != (n_samples, dim, dim):
            raise ValueError(f"{name} must have shape ({n_samples}, {dim}, {dim})")
        return covariance_array
    raise ValueError(f"{name} must have shape (dim, dim) or (n, dim, dim)")


def _quadratic_forms(vectors: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    solved = np.linalg.solve(matrices, vectors[..., np.newaxis])[..., 0]
    return np.sum(vectors * solved, axis=1)


def _validate_order_cutoff(order: float, cutoff: float) -> tuple[float, float]:
    order = float(order)
    cutoff = float(cutoff)
    if order < 1.0:
        raise ValueError("order must be at least 1")
    if cutoff <= 0.0 or not np.isfinite(cutoff):
        raise ValueError("cutoff must be a finite positive number")
    return order, cutoff


def _coerce_point_sets(points1: ArrayLike, points2: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    first = _coerce_point_set(points1, "estimated_points")
    second = _coerce_point_set(points2, "reference_points")
    if first.shape[0] and second.shape[0] and first.shape[1] != second.shape[1]:
        raise ValueError("point sets must have the same point dimension")
    return first, second


def _coerce_point_set(points: ArrayLike, name: str) -> np.ndarray:
    points_np = _as_numeric_array(points, name)
    if points_np.size == 0:
        return points_np.reshape(0, 0)
    if points_np.ndim == 1:
        return points_np.reshape(-1, 1)
    if points_np.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array-like point set")
    return points_np


def _clipped_assignment_cost(estimated: np.ndarray, reference: np.ndarray, *, order: float, cutoff: float, distance_fn: DistanceFunction | None) -> tuple[float, list[tuple[int, int]]]:
    if estimated.shape[0] == 0 or reference.shape[0] == 0:
        return 0.0, []
    distances = _pairwise_distances(estimated, reference, distance_fn)
    clipped_costs = np.minimum(distances, cutoff) ** order
    row_ind, col_ind = linear_sum_assignment(clipped_costs)
    assignment_cost = float(clipped_costs[row_ind, col_ind].sum())
    return assignment_cost, [(int(row), int(col)) for row, col in zip(row_ind, col_ind)]


def _pairwise_distances(estimated: np.ndarray, reference: np.ndarray, distance_fn: DistanceFunction | None) -> np.ndarray:
    if distance_fn is None:
        differences = estimated[:, None, :] - reference[None, :, :]
        return np.linalg.norm(differences, axis=-1)
    distances = np.empty((estimated.shape[0], reference.shape[0]), dtype=float)
    for row, estimate in enumerate(estimated):
        for col, truth in enumerate(reference):
            distances[row, col] = float(distance_fn(estimate, truth))
    return distances


def _components_or_distance(name: str, distance: float, localization_component: float, cardinality_component: float, assignments: int, return_components: bool) -> float | dict[str, float | int]:
    if not return_components:
        return float(distance)
    return {
        name: float(distance),
        "distance": float(distance),
        "localization": float(localization_component),
        "cardinality": float(cardinality_component),
        "assignments": int(assignments),
    }


def _validate_square_matrix(matrix: np.ndarray, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _symmetric_matrix_square_root(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetrize(matrix))
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
