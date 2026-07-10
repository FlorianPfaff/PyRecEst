"""Shared helpers for rigid and non-rigid point-set registration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    array_equal,
    asarray,
    cast,
    empty,
    full,
    int64,
    isfinite,
    mean,
    sqrt,
    where,
    zeros,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class RegistrationResultBase:  # pylint: disable=too-many-instance-attributes
    """Shared fields for registration result containers."""

    assignment: Any
    matched_reference_indices: Any
    matched_moving_indices: Any
    transformed_reference_points: Any
    matched_costs: Any
    rmse: float
    n_iterations: int
    converged: bool


@dataclass(frozen=True)
class MatchSummary:
    """Matched indices, costs, and RMSE diagnostics for an assignment."""

    matched_reference_indices: Any
    matched_moving_indices: Any
    matched_costs: Any
    rmse: float


@dataclass(frozen=True)
class RegistrationLoopState:
    """Common bookkeeping state for alternating registration loops."""

    transform: Any
    assignment: Any
    transformed_reference_points: Any
    costs: Any
    iteration: int
    converged: bool


@dataclass(frozen=True)
class RegistrationLoopConfig:
    """Configuration shared by the rigid and non-rigid registration loops."""

    max_cost: float
    cost_function: Any
    max_iterations: int
    min_matches: int
    tolerance: float


@dataclass(frozen=True)
class RegistrationLoopCallbacks:
    """Problem-specific callbacks for the shared registration loop."""

    fit_transform: Callable[[Any, Any], Any]
    compute_change: Callable[[Any, Any, Any, Any], float]
    assignment_solver: Callable[..., Any]


def as_point_array(
    points,
    *,
    expected_dim: int | None = None,
    expected_dim_error: str | None = None,
):
    """Validate a point array and optionally enforce its ambient dimension."""
    point_array = asarray(points)
    if point_array.ndim != 2:
        raise ValueError("points must have shape (n_points, dim).")
    if point_array.shape[0] == 0:
        raise ValueError("At least one point is required.")
    if (
        _is_boolean_dtype(point_array)
        or _is_complex_dtype(point_array)
        or _is_text_dtype(point_array)
        or _is_temporal_dtype(point_array)
    ):
        raise ValueError("points must contain real numeric values.")
    if not bool(backend_all(isfinite(point_array))):
        raise ValueError("points must contain only finite values.")
    if expected_dim is None:
        if point_array.shape[1] == 0:
            raise ValueError("Point dimension must be positive.")
    elif point_array.shape[1] != expected_dim:
        raise ValueError(
            expected_dim_error or f"points must have dimension {expected_dim}."
        )
    return point_array


def validate_pair(
    source_points,
    target_points,
    *,
    expected_dim: int | None = None,
    expected_dim_error: str | None = None,
):
    """Validate a pair of matched point arrays."""
    source = as_point_array(
        source_points,
        expected_dim=expected_dim,
        expected_dim_error=expected_dim_error,
    )
    target = as_point_array(
        target_points,
        expected_dim=expected_dim,
        expected_dim_error=expected_dim_error,
    )
    if source.shape != target.shape:
        raise ValueError("source_points and target_points must have the same shape.")
    return source, target


def _dtype_kind(value) -> str | None:
    dtype = getattr(value, "dtype", None)
    return getattr(dtype, "kind", None)


def _dtype_name(value) -> str:
    return str(getattr(value, "dtype", "")).lower()


def _is_boolean_dtype(value) -> bool:
    return _dtype_kind(value) == "b" or _dtype_name(value) in {
        "bool",
        "bool_",
        "torch.bool",
    }


def _is_complex_dtype(value) -> bool:
    return _dtype_kind(value) == "c" or _dtype_name(value) in {
        "complex64",
        "complex128",
        "complex256",
        "torch.complex64",
        "torch.complex128",
    }


def _is_text_dtype(value) -> bool:
    return _dtype_kind(value) in {"S", "U"} or _dtype_name(value) in {
        "bytes",
        "str",
        "string",
    }


def _is_temporal_dtype(value) -> bool:
    dtype_kind = _dtype_kind(value)
    dtype_name = _dtype_name(value)
    return dtype_kind in {"M", "m"} or any(
        temporal_name in dtype_name for temporal_name in ("datetime64", "timedelta64")
    )


def _is_numpy_temporal_scalar(value) -> bool:
    value_type = type(value)
    return value_type.__module__ == "numpy" and value_type.__name__ in {
        "datetime64",
        "timedelta64",
    }


def _coerce_cost_matrix(cost_matrix):
    try:
        costs = asarray(cost_matrix)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError("cost_matrix must contain real numeric values.") from exc

    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional.")
    if (
        _is_boolean_dtype(costs)
        or _is_complex_dtype(costs)
        or _is_text_dtype(costs)
        or _is_temporal_dtype(costs)
    ):
        raise ValueError("cost_matrix must contain real numeric values.")

    try:
        finite_mask = isfinite(costs)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError("cost_matrix must contain real numeric values.") from exc
    return costs, finite_mask


def validate_cost_matrix(cost_matrix, *, n_reference: int, n_moving: int):
    """Validate the shape and numeric contract of an association cost matrix."""
    costs, _ = _coerce_cost_matrix(cost_matrix)
    if costs.shape != (n_reference, n_moving):
        raise ValueError(
            "cost_function must return an array of shape (n_reference, n_moving)."
        )
    return costs


def evaluate_registration_costs(transform, reference, moving, association_cost):
    """Transform reference points and evaluate the association cost matrix."""
    transformed_reference = transform.apply(reference)
    current_costs = validate_cost_matrix(
        association_cost(transformed_reference, moving),
        n_reference=reference.shape[0],
        n_moving=moving.shape[0],
    )
    return transformed_reference, current_costs


def _validate_max_cost(max_cost) -> float:
    try:
        max_cost_array = asarray(max_cost)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError("max_cost must be a scalar numeric value.") from exc
    if max_cost_array.shape != ():
        raise ValueError("max_cost must be a scalar.")
    if _is_temporal_dtype(max_cost_array):
        raise ValueError("max_cost must be a scalar numeric value.")

    try:
        max_cost_scalar = max_cost_array.item()
    except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
        raise ValueError("max_cost must be a scalar numeric value.") from exc
    if isinstance(
        max_cost_scalar, (bool, str, bytes, bytearray)
    ) or _is_numpy_temporal_scalar(max_cost_scalar):
        raise ValueError("max_cost must be a scalar numeric value.")

    try:
        max_cost_value = float(max_cost_scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_cost must be a scalar numeric value.") from exc
    if math.isnan(max_cost_value) or max_cost_value < 0.0:
        raise ValueError("max_cost must be non-negative or infinity.")
    return max_cost_value


def _validate_tolerance(tolerance) -> float:
    if isinstance(tolerance, (str, bytes, bytearray)):
        raise ValueError("tolerance must be a finite non-negative scalar.")

    try:
        tolerance_array = asarray(tolerance)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError("tolerance must be a finite non-negative scalar.") from exc
    if tolerance_array.shape != ():
        raise ValueError("tolerance must be a finite non-negative scalar.")
    if _is_temporal_dtype(tolerance_array):
        raise ValueError("tolerance must be a finite non-negative scalar.")

    tolerance_scalar = tolerance_array.item()
    if isinstance(
        tolerance_scalar, (bool, str, bytes, bytearray)
    ) or _is_numpy_temporal_scalar(tolerance_scalar):
        raise ValueError("tolerance must be a finite non-negative scalar.")

    try:
        tolerance_value = float(tolerance_scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tolerance must be a finite non-negative scalar.") from exc

    if not math.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError("tolerance must be a finite non-negative scalar.")
    return tolerance_value


def solve_gated_assignment(cost_matrix, *, max_cost: float = float("inf")):
    """Solve one-to-one assignment with optional gating."""
    costs, finite_mask = _coerce_cost_matrix(cost_matrix)
    if costs.shape[0] == 0:
        return zeros((0,), dtype=int64)
    if costs.shape[1] == 0:
        return zeros((costs.shape[0],), dtype=int64) - 1

    max_cost_value = _validate_max_cost(max_cost)
    valid_cost_mask = finite_mask
    if math.isfinite(max_cost_value):
        valid_cost_mask = finite_mask & (costs <= max_cost_value)
    valid_costs = costs[valid_cost_mask]
    if valid_costs.shape[0] == 0:
        return zeros((costs.shape[0],), dtype=int64) - 1

    highest_valid_cost = float(valid_costs.max())
    unassigned_cost = (
        max_cost_value if math.isfinite(max_cost_value) else highest_valid_cost + 1.0
    )
    cost_scale = max(abs(highest_valid_cost), abs(unassigned_cost), 1.0)
    invalid_cost = max(highest_valid_cost, 2.0 * unassigned_cost) + cost_scale

    n_reference, n_moving = costs.shape
    augmented_size = n_reference + n_moving
    augmented_costs = full((augmented_size, augmented_size), invalid_cost)
    augmented_costs[:n_reference, :n_moving] = where(
        valid_cost_mask,
        costs,
        invalid_cost,
    )

    for reference_index in range(n_reference):
        augmented_costs[reference_index, n_moving + reference_index] = unassigned_cost
    for moving_index in range(n_moving):
        augmented_costs[n_reference + moving_index, moving_index] = unassigned_cost
    augmented_costs[n_reference:, n_moving:] = 0.0

    row_indices, col_indices = linear_sum_assignment(augmented_costs)

    assignment = zeros((n_reference,), dtype=int64) - 1
    for row_index, col_index in zip(row_indices, col_indices):
        if row_index >= n_reference or col_index >= n_moving:
            continue
        if bool(valid_cost_mask[row_index, col_index]):
            assignment[row_index] = int(col_index)
    return assignment


def default_cost(transformed_reference_points, moving_points):
    """Default Euclidean association cost."""
    return cdist(transformed_reference_points, moving_points, metric="euclidean")


def compute_rmse(matched_costs) -> float:
    """Compute the RMSE over matched costs."""
    if matched_costs.shape[0] > 0:
        return float(sqrt(mean(matched_costs * matched_costs)))
    return float("inf")


def summarize_assignment(assignment, costs) -> MatchSummary:
    """Extract matched row/column indices and diagnostics from an assignment."""
    matched_reference_indices = where(assignment >= 0)[0]
    matched_moving_indices = assignment[matched_reference_indices]
    matched_costs = (
        costs[matched_reference_indices, matched_moving_indices]
        if matched_reference_indices.shape[0] > 0
        else empty((0,))
    )
    return MatchSummary(
        matched_reference_indices=cast(matched_reference_indices, int64),
        matched_moving_indices=cast(matched_moving_indices, int64),
        matched_costs=matched_costs,
        rmse=compute_rmse(matched_costs),
    )


def run_registration_loop(  # pylint: disable=too-many-locals
    reference,
    moving,
    initial_transform,
    config: RegistrationLoopConfig,
    callbacks: RegistrationLoopCallbacks,
) -> RegistrationLoopState:
    """Run the shared alternating assignment/refit loop."""
    transform = initial_transform
    assignment = zeros((reference.shape[0],), dtype=int64) - 1
    converged = False
    iteration = 0
    tolerance = _validate_tolerance(config.tolerance)
    association_cost = (
        default_cost if config.cost_function is None else config.cost_function
    )

    for iteration in range(1, config.max_iterations + 1):
        transformed_reference, current_costs = evaluate_registration_costs(
            transform,
            reference,
            moving,
            association_cost,
        )

        new_assignment = callbacks.assignment_solver(
            current_costs,
            max_cost=config.max_cost,
        )
        matched_reference_indices = where(new_assignment >= 0)[0]

        if matched_reference_indices.shape[0] < config.min_matches:
            return RegistrationLoopState(
                transform=transform,
                assignment=new_assignment,
                transformed_reference_points=transformed_reference,
                costs=current_costs,
                iteration=iteration,
                converged=False,
            )

        matched_moving_indices = new_assignment[matched_reference_indices]
        updated_transform = callbacks.fit_transform(
            reference[matched_reference_indices],
            moving[matched_moving_indices],
        )

        change = callbacks.compute_change(
            transform,
            updated_transform,
            reference,
            transformed_reference,
        )
        same_assignment = bool(array_equal(new_assignment, assignment))

        transform = updated_transform
        assignment = new_assignment

        if same_assignment and change <= tolerance:
            converged = True
            break

    transformed_reference, final_costs = evaluate_registration_costs(
        transform,
        reference,
        moving,
        association_cost,
    )
    final_assignment = callbacks.assignment_solver(
        final_costs,
        max_cost=config.max_cost,
    )
    if converged and not bool(array_equal(final_assignment, assignment)):
        converged = False
    return RegistrationLoopState(
        transform=transform,
        assignment=final_assignment,
        transformed_reference_points=transformed_reference,
        costs=final_costs,
        iteration=iteration,
        converged=converged,
    )


def build_registration_result(result_type, state: RegistrationLoopState):
    """Construct a registration result object from common bookkeeping."""
    summary = summarize_assignment(state.assignment, state.costs)
    return result_type(
        transform=state.transform,
        assignment=state.assignment,
        matched_reference_indices=summary.matched_reference_indices,
        matched_moving_indices=summary.matched_moving_indices,
        transformed_reference_points=state.transformed_reference_points,
        matched_costs=summary.matched_costs,
        rmse=summary.rmse,
        n_iterations=state.iteration,
        converged=state.converged,
    )
