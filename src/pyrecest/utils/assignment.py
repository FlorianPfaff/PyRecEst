"""Utilities for k-best partial linear assignments."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

import pyrecest.backend
from pyrecest.backend import abs as _abs
from pyrecest.backend import any as _any
from pyrecest.backend import array as _array
from pyrecest.backend import asarray as _asarray
from pyrecest.backend import concatenate as _concatenate
from pyrecest.backend import full as _full
from pyrecest.backend import int64 as _int64
from pyrecest.backend import isfinite as _isfinite
from pyrecest.backend import sum as _sum
from pyrecest.backend import where as _where
from pyrecest.backend import zeros as _zeros
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class _MurtySubproblem:
    """Internal Murty subproblem descriptor."""

    forced_pairs: tuple[tuple[int, int], ...]
    forbidden_pairs: tuple[tuple[int, int], ...]
    branching_row_start: int


def _coerce_non_assignment_costs(costs, size: int, name: str):
    if costs is None:
        return _zeros(size, dtype=float)

    costs = _asarray(costs, dtype=float).reshape(-1)
    if costs.shape[0] != size:
        raise ValueError(f"{name} must have length {size}")
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

    forced_rows = set()
    forced_cols = set()
    for row_index, col_index in subproblem.forced_pairs:
        if row_index in forced_rows or col_index in forced_cols:
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
    """Compute the k best one-to-one partial assignments.

    Parameters
    ----------
    cost_matrix : array_like, shape (n_rows, n_cols)
        Matrix containing the cost of assigning each row to each column.
        Forbidden assignments can be encoded as ``numpy.inf``.
    k : int, default=1
        Number of ranked assignments to return.
    row_non_assignment_costs : array_like, optional
        Cost incurred when a row remains unassigned. If omitted, zero cost is
        used for all rows.
    col_non_assignment_costs : array_like, optional
        Cost incurred when a column remains unassigned. If omitted, zero cost is
        used for all columns.

    Returns
    -------
    list[dict]
        Ranked assignment solutions in ascending cost order. Each dictionary has
        the keys ``assignment`` (matched column per row or ``-1``),
        ``unassigned_rows``, ``unassigned_cols``, and ``cost``.

    Notes
    -----
    The implementation follows Murty's partitioning strategy on an augmented
    square assignment formulation. Branching is performed only over the original
    rows, which directly yields unique ranked partial assignments.
    """
    if k <= 0:
        return []

    if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
        raise NotImplementedError(
            "murty_k_best_assignments is not supported on the JAX backend."
        )

    cost_matrix = _asarray(cost_matrix, dtype=float)
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be a two-dimensional array")

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


__all__ = ["murty_k_best_assignments"]
