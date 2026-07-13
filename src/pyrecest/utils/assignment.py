"""Utilities for k-best partial linear assignments."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import isfinite as _is_scalar_finite
from numbers import Integral

import numpy as _np
import pyrecest.backend
from pyrecest.backend import abs as _abs
from pyrecest.backend import any as _any
from pyrecest.backend import amax as _amax
from pyrecest.backend import array as _array
from pyrecest.backend import asarray as _asarray
from pyrecest.backend import concatenate as _concatenate
from pyrecest.backend import full as _full
from pyrecest.backend import int64 as _int64
from pyrecest.backend import isfinite as _isfinite
from pyrecest.backend import isinf as _isinf
from pyrecest.backend import isnan as _isnan
from pyrecest.backend import sum as _sum
from pyrecest.backend import where as _where
from pyrecest.backend import zeros as _zeros
from scipy.optimize import linear_sum_assignment

_TEXT_TYPES = (str, bytes, bytearray, _np.str_, _np.bytes_)
_BOOLEAN_TYPES = (bool, _np.bool_)
_COMPLEX_TYPES = (complex, _np.complexfloating)
_TEMPORAL_TYPES = (_np.datetime64, _np.timedelta64)
_INVALID_SCALAR_TYPES = _BOOLEAN_TYPES + _TEXT_TYPES + _TEMPORAL_TYPES


@dataclass(frozen=True)
class _MurtySubproblem:
    """Internal Murty subproblem descriptor."""

    forced_pairs: tuple[tuple[int, int], ...]
    forbidden_pairs: tuple[tuple[int, int], ...]
    branching_row_start: int


def _validate_assignment_count(k: int) -> int:
    if isinstance(k, _INVALID_SCALAR_TYPES):
        raise ValueError("k must be an integer")
    if isinstance(k, Integral):
        return int(k)
    if not (hasattr(k, "ndim") or hasattr(k, "shape") or hasattr(k, "item")):
        raise ValueError("k must be an integer")

    try:
        k_array = _asarray(k)
    except (TypeError, ValueError) as exc:
        raise ValueError("k must be an integer") from exc
    if k_array.ndim != 0 or _has_temporal_dtype(k_array):
        raise ValueError("k must be an integer")

    try:
        scalar = k_array.item()
    except AttributeError:
        scalar = k_array
    if isinstance(scalar, _INVALID_SCALAR_TYPES):
        raise ValueError("k must be an integer")

    try:
        scalar_float = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("k must be an integer") from exc
    if not _is_scalar_finite(scalar_float) or not scalar_float.is_integer():
        raise ValueError("k must be an integer")
    return int(scalar_float)


def _has_boolean_dtype(value) -> bool:
    dtype = getattr(value, "dtype", None)
    return dtype is not None and str(dtype).lower() in {"bool", "bool_", "torch.bool"}


def _has_complex_dtype(value) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    try:
        return bool(_np.issubdtype(dtype, _np.complexfloating))
    except TypeError:
        return str(dtype).lower() in {
            "complex64",
            "complex128",
            "torch.complex64",
            "torch.complex128",
        }


def _has_temporal_dtype(value) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    try:
        return bool(
            _np.issubdtype(dtype, _np.datetime64)
            or _np.issubdtype(dtype, _np.timedelta64)
        )
    except TypeError:
        dtype_name = str(dtype).lower()
        return "datetime64" in dtype_name or "timedelta64" in dtype_name


def _contains_boolean_values(value) -> bool:
    if isinstance(value, _BOOLEAN_TYPES):
        return True
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _BOOLEAN_TYPES) for item in values)


def _contains_text_values(value) -> bool:
    if isinstance(value, _TEXT_TYPES):
        return True
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _TEXT_TYPES) for item in values)


def _contains_temporal_values(value) -> bool:
    if isinstance(value, _TEMPORAL_TYPES):
        return True
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _TEMPORAL_TYPES) for item in values)


def _contains_complex_values(value) -> bool:
    if isinstance(value, _COMPLEX_TYPES):
        return True
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _COMPLEX_TYPES) for item in values)


def _coerce_cost_matrix(cost_matrix):
    if _contains_boolean_values(cost_matrix):
        raise ValueError("cost_matrix must be numeric, not boolean")
    if _contains_text_values(cost_matrix) or _contains_temporal_values(cost_matrix):
        raise ValueError("cost_matrix must be numeric")
    if _contains_complex_values(cost_matrix):
        raise ValueError("cost_matrix must be real-valued")
    try:
        raw_cost_matrix = _asarray(cost_matrix)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError("cost_matrix must be numeric") from exc
    if _has_boolean_dtype(raw_cost_matrix):
        raise ValueError("cost_matrix must be numeric, not boolean")
    if _contains_boolean_values(cost_matrix) or _contains_boolean_values(
        raw_cost_matrix
    ):
        raise ValueError("cost_matrix must be numeric, not boolean")
    if (
        _contains_text_values(cost_matrix)
        or _contains_text_values(raw_cost_matrix)
        or _has_temporal_dtype(raw_cost_matrix)
        or _contains_temporal_values(raw_cost_matrix)
    ):
        raise ValueError("cost_matrix must be numeric")
    if _has_complex_dtype(raw_cost_matrix) or _contains_complex_values(raw_cost_matrix):
        raise ValueError("cost_matrix must be real-valued")

    try:
        coerced_cost_matrix = _asarray(raw_cost_matrix, dtype=float)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError("cost_matrix must be numeric") from exc
    if coerced_cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be a two-dimensional array")
    if _any(_isnan(coerced_cost_matrix)) or _any(
        _isinf(coerced_cost_matrix) & (coerced_cost_matrix < 0.0)
    ):
        raise ValueError(
            "cost_matrix may only contain finite values or positive infinity"
        )
    return coerced_cost_matrix


def _coerce_non_assignment_costs(costs, size: int, name: str):
    if costs is None:
        return _zeros(size, dtype=float)

    if _contains_temporal_values(costs):
        raise ValueError(f"{name} must be numeric and finite")
    if _contains_complex_values(costs):
        raise ValueError(f"{name} must be real-valued")
    try:
        raw_costs_array = _asarray(costs)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError(f"{name} must be numeric and finite") from exc
    if _has_boolean_dtype(raw_costs_array):
        raise ValueError(f"{name} must be numeric and finite")
    if _contains_boolean_values(costs) or _contains_boolean_values(raw_costs_array):
        raise ValueError(f"{name} must be numeric and finite")
    if (
        _contains_text_values(costs)
        or _contains_text_values(raw_costs_array)
        or _has_temporal_dtype(raw_costs_array)
        or _contains_temporal_values(raw_costs_array)
    ):
        raise ValueError(f"{name} must be numeric and finite")
    if _has_complex_dtype(raw_costs_array) or _contains_complex_values(raw_costs_array):
        raise ValueError(f"{name} must be real-valued")

    try:
        costs_array = _asarray(raw_costs_array, dtype=float)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError(f"{name} must be numeric and finite") from exc
    if costs_array.ndim == 0:
        try:
            cost = float(costs_array)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be numeric and finite") from exc
        if not _is_scalar_finite(cost):
            raise ValueError(f"{name} must be finite")
        return _full((size,), cost, dtype=float)

    costs = costs_array.reshape(-1)
    if costs.shape[0] != size:
        raise ValueError(f"{name} must be scalar or have length {size}")
    if _any(~_isfinite(costs)):
        raise ValueError(f"{name} must be finite")
    return costs


def _get_large_cost(cost_matrix, row_non_assignment_costs, col_non_assignment_costs):
    finite_costs = cost_matrix[_isfinite(cost_matrix)]
    finite_entries = _concatenate(
        (
            finite_costs.reshape(-1),
            row_non_assignment_costs.reshape(-1),
            col_non_assignment_costs.reshape(-1),
        )
    )
    return 2.0 * (float(_sum(_abs(finite_entries))) + 1.0)


def _build_augmented_cost_matrix(
    cost_matrix,
    row_non_assignment_costs,
    col_non_assignment_costs,
    large_cost,
):
    n_rows, n_cols = cost_matrix.shape
    augmented_cost_matrix = _full(
        (n_rows + n_cols, n_cols + n_rows),
        large_cost,
        dtype=float,
    )

    if n_rows > 0 and n_cols > 0:
        finite_cost_matrix = _array(cost_matrix)
        finite_cost_matrix[~_isfinite(finite_cost_matrix)] = large_cost
        augmented_cost_matrix[:n_rows, :n_cols] = finite_cost_matrix

    for row_index, row_cost in enumerate(row_non_assignment_costs):
        augmented_cost_matrix[row_index, n_cols + row_index] = row_cost

    for col_index, col_cost in enumerate(col_non_assignment_costs):
        augmented_cost_matrix[n_rows + col_index, col_index] = col_cost

    augmented_cost_matrix[n_rows:, n_cols:] = 0.0
    return augmented_cost_matrix


def _solve_subproblem(  # pylint: disable=too-many-locals
    augmented_cost_matrix,
    n_rows: int,
    n_cols: int,
    large_cost: float,
    subproblem: _MurtySubproblem,
):
    modified_cost_matrix = _array(augmented_cost_matrix)

    for row_index, col_index in subproblem.forbidden_pairs:
        modified_cost_matrix[row_index, col_index] = large_cost

    forbidden_pairs = set(subproblem.forbidden_pairs)
    forced_rows = set()
    forced_cols = set()
    for row_index, col_index in subproblem.forced_pairs:
        if (
            row_index in forced_rows
            or col_index in forced_cols
            or (row_index, col_index) in forbidden_pairs
        ):
            return None
        if bool(augmented_cost_matrix[row_index, col_index] >= large_cost / 2.0):
            return None
        forced_rows.add(row_index)
        forced_cols.add(col_index)

    for row_index, col_index in subproblem.forced_pairs:
        modified_cost_matrix[row_index, :] = large_cost
        modified_cost_matrix[:, col_index] = large_cost
        modified_cost_matrix[row_index, col_index] = augmented_cost_matrix[
            row_index, col_index
        ]

    row_ind, col_ind = linear_sum_assignment(modified_cost_matrix)
    chosen_costs = modified_cost_matrix[row_ind, col_ind]
    if _any(chosen_costs >= large_cost / 2.0):
        return None

    full_assignment = _full((n_rows,), -1, dtype=_int64)
    for row_index, col_index in zip(row_ind, col_ind):
        if row_index < n_rows:
            full_assignment[row_index] = col_index

    assignment = _full((n_rows,), -1, dtype=_int64)
    for row_index, col_index in enumerate(full_assignment):
        if col_index < n_cols:
            assignment[row_index] = col_index

    assigned_columns = {int(col_index) for col_index in assignment if col_index >= 0}
    unassigned_rows = _asarray(_where(assignment < 0)[0], dtype=_int64)
    unassigned_cols = _asarray(
        [col_index for col_index in range(n_cols) if col_index not in assigned_columns],
        dtype=_int64,
    )

    total_cost = float(augmented_cost_matrix[row_ind, col_ind].sum())
    return {
        "assignment": assignment,
        "unassigned_rows": unassigned_rows,
        "unassigned_cols": unassigned_cols,
        "cost": total_cost,
        "_full_assignment": full_assignment,
    }


def murty_k_best_assignments(  # pylint: disable=too-many-locals
    cost_matrix,
    k: int = 1,
    row_non_assignment_costs=None,
    col_non_assignment_costs=None,
):
    """Compute the k best one-to-one partial assignments."""
    k = _validate_assignment_count(k)
    if k <= 0:
        return []

    if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
        raise NotImplementedError(
            "murty_k_best_assignments is not supported on the JAX backend."
        )

    cost_matrix = _coerce_cost_matrix(cost_matrix)

    n_rows, n_cols = cost_matrix.shape
    row_non_assignment_costs = _coerce_non_assignment_costs(
        row_non_assignment_costs,
        n_rows,
        "row_non_assignment_costs",
    )
    col_non_assignment_costs = _coerce_non_assignment_costs(
        col_non_assignment_costs,
        n_cols,
        "col_non_assignment_costs",
    )

    large_cost = _get_large_cost(
        cost_matrix,
        row_non_assignment_costs,
        col_non_assignment_costs,
    )
    augmented_cost_matrix = _build_augmented_cost_matrix(
        cost_matrix,
        row_non_assignment_costs,
        col_non_assignment_costs,
        large_cost,
    )

    root_subproblem = _MurtySubproblem(tuple(), tuple(), 0)
    root_solution = _solve_subproblem(
        augmented_cost_matrix,
        n_rows,
        n_cols,
        large_cost,
        root_subproblem,
    )
    if root_solution is None:
        return []

    solution_heap: list[tuple[float, int, _MurtySubproblem, dict]] = []
    counter = 0
    heappush(
        solution_heap,
        (root_solution["cost"], counter, root_subproblem, root_solution),
    )
    counter += 1

    ranked_solutions: list[dict] = []
    while solution_heap and len(ranked_solutions) < k:
        _, _, subproblem, solution = heappop(solution_heap)
        ranked_solutions.append(
            {
                "assignment": solution["assignment"],
                "unassigned_rows": solution["unassigned_rows"],
                "unassigned_cols": solution["unassigned_cols"],
                "cost": solution["cost"],
            }
        )

        forced_prefix = list(subproblem.forced_pairs)
        for row_index in range(subproblem.branching_row_start, n_rows):
            child_subproblem = _MurtySubproblem(
                tuple(forced_prefix),
                subproblem.forbidden_pairs
                + ((row_index, int(solution["_full_assignment"][row_index])),),
                row_index,
            )
            child_solution = _solve_subproblem(
                augmented_cost_matrix,
                n_rows,
                n_cols,
                large_cost,
                child_subproblem,
            )
            if child_solution is not None:
                heappush(
                    solution_heap,
                    (
                        child_solution["cost"],
                        counter,
                        child_subproblem,
                        child_solution,
                    ),
                )
                counter += 1

            forced_prefix.append(
                (row_index, int(solution["_full_assignment"][row_index]))
            )

    return ranked_solutions


def min_cost_max_cardinality_assignment(cost_matrix):
    """Compute the cheapest assignment among maximum-cardinality matchings."""
    if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
        raise NotImplementedError(
            "min_cost_max_cardinality_assignment is not supported on the JAX backend."
        )

    cost_matrix = _coerce_cost_matrix(cost_matrix)

    n_rows, n_cols = cost_matrix.shape
    assignment = _full((n_rows,), -1, dtype=_int64)
    finite_costs = cost_matrix[_isfinite(cost_matrix)]
    if finite_costs.shape[0] == 0:
        return {
            "assignment": assignment,
            "unassigned_rows": _asarray(list(range(n_rows)), dtype=_int64),
            "unassigned_cols": _asarray(list(range(n_cols)), dtype=_int64),
            "cost": 0.0,
        }

    finite_cost_mask = _isfinite(cost_matrix)
    optimization_cost_matrix = _array(cost_matrix)
    maximum_absolute_cost = float(_amax(_abs(finite_costs)))
    if maximum_absolute_cost > 0.0:
        optimization_cost_matrix[finite_cost_mask] /= maximum_absolute_cost
    scaled_finite_costs = optimization_cost_matrix[finite_cost_mask]
    cardinality_priority_cost = 2.0 * (
        float(_sum(_abs(scaled_finite_costs))) + 1.0
    )
    solutions = murty_k_best_assignments(
        optimization_cost_matrix,
        k=1,
        row_non_assignment_costs=_full(
            (n_rows,), cardinality_priority_cost, dtype=float
        ),
        col_non_assignment_costs=_zeros(n_cols, dtype=float),
    )
    if not solutions:
        return {
            "assignment": assignment,
            "unassigned_rows": _asarray(list(range(n_rows)), dtype=_int64),
            "unassigned_cols": _asarray(list(range(n_cols)), dtype=_int64),
            "cost": 0.0,
        }

    solution = solutions[0]
    assignment = solution["assignment"]
    assigned_rows = _where(assignment >= 0)[0]
    total_cost = 0.0
    if assigned_rows.shape[0] > 0:
        total_cost = float(_sum(cost_matrix[assigned_rows, assignment[assigned_rows]]))
    return {
        "assignment": assignment,
        "unassigned_rows": solution["unassigned_rows"],
        "unassigned_cols": solution["unassigned_cols"],
        "cost": total_cost,
    }


__all__ = ["min_cost_max_cardinality_assignment", "murty_k_best_assignments"]
