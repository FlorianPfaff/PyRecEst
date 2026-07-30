"""Runtime patch for finite-range ROI similarity assignment costs."""

from __future__ import annotations

import math
import sys


def _cost_conversion_is_finite(
    max_similarity: float,
    min_similarity: float,
) -> bool:
    """Return whether the existing score-to-cost conversion stays finite."""

    threshold_cost = max_similarity - min_similarity
    dummy_penalty = max(
        1e-12,
        sys.float_info.epsilon * max(1.0, abs(max_similarity), abs(min_similarity)),
    )
    return math.isfinite(threshold_cost) and math.isfinite(
        threshold_cost + dummy_penalty
    )


def _stable_extreme_assignment(
    roi_assignment_module,
    similarities,
    *,
    minimum: float,
    num_dummy,
    unmatched_value: int,
):
    """Solve an unsafe finite-range assignment without collapsing score ordering."""

    n_rows, n_cols = similarities.shape
    if num_dummy is None:
        num_dummy = max(n_rows, n_cols)
    else:
        num_dummy = roi_assignment_module._as_nonnegative_integer(
            num_dummy,
            "num_dummy",
        )

    finite_mask = roi_assignment_module.isfinite(similarities)
    valid_mask = finite_mask & (similarities >= minimum)
    maximum = float(roi_assignment_module.amax(similarities[finite_mask]))
    padded_size = max(n_rows, n_cols) + num_dummy

    dummy_similarity = math.nextafter(minimum, -math.inf)
    if math.isfinite(dummy_similarity):
        score_matrix = roi_assignment_module.full_like(
            similarities,
            dummy_similarity,
        )
        score_matrix[valid_mask] = similarities[valid_mask]
        padded_matrix = roi_assignment_module.full(
            (padded_size, padded_size),
            dummy_similarity,
            dtype=roi_assignment_module.float64,
        )
        padded_matrix[:n_rows, :n_cols] = score_matrix
        row_ind, col_ind = roi_assignment_module.linear_sum_assignment(
            padded_matrix,
            maximize=True,
        )
    else:
        # The smallest finite float has no finite predecessor for dummy matches.
        # Retain the normalized cost fallback for that boundary-only case.
        scale = max(1.0, abs(maximum), abs(minimum))
        normalized_similarities = similarities / scale
        normalized_maximum = maximum / scale
        normalized_minimum = minimum / scale

        threshold_cost = normalized_maximum - normalized_minimum
        dummy_penalty = max(1e-12 / scale, sys.float_info.epsilon)
        dummy_cost = threshold_cost + dummy_penalty
        if dummy_cost <= threshold_cost:
            dummy_cost = math.nextafter(threshold_cost, math.inf)

        cost_matrix = roi_assignment_module.full_like(
            normalized_similarities,
            dummy_cost,
        )
        cost_matrix[valid_mask] = (
            normalized_maximum - normalized_similarities[valid_mask]
        )
        padded_matrix = roi_assignment_module.full(
            (padded_size, padded_size),
            dummy_cost,
            dtype=roi_assignment_module.float64,
        )
        padded_matrix[:n_rows, :n_cols] = cost_matrix
        row_ind, col_ind = roi_assignment_module.linear_sum_assignment(padded_matrix)

    assignment = roi_assignment_module.full(
        (n_rows,),
        unmatched_value,
        dtype=roi_assignment_module.int64,
    )
    for row_index, col_index in zip(row_ind, col_ind):
        if (
            row_index < n_rows
            and col_index < n_cols
            and valid_mask[row_index, col_index]
        ):
            assignment[row_index] = int(col_index)
    return assignment


def patch_similarity_assignment_extreme_range(roi_assignment_module) -> None:
    """Normalize only score ranges that overflow the Hungarian cost transform."""

    storage_name = "_assign_by_similarity_matrix_without_unmatched_value_validation"
    current = getattr(
        roi_assignment_module,
        storage_name,
        roi_assignment_module.assign_by_similarity_matrix,
    )
    if getattr(current, "_pyrecest_finite_similarity_cost_range", False):
        return

    original_assign = current

    # pylint: disable=too-many-return-statements
    def assign_by_similarity_matrix(
        similarity_matrix,
        min_similarity=0.0,
        num_dummy=None,
        unmatched_value=-1,
        *,
        return_result=False,
    ):
        """Solve a one-to-one assignment problem by maximizing similarity."""

        if roi_assignment_module.pyrecest.backend.__backend_name__ == "jax":
            return original_assign(
                similarity_matrix,
                min_similarity=min_similarity,
                num_dummy=num_dummy,
                unmatched_value=unmatched_value,
                return_result=return_result,
            )

        similarities = roi_assignment_module.asarray(
            similarity_matrix,
            dtype=roi_assignment_module.float64,
        )
        if similarities.ndim != 2:
            return original_assign(
                similarity_matrix,
                min_similarity=min_similarity,
                num_dummy=num_dummy,
                unmatched_value=unmatched_value,
                return_result=return_result,
            )

        try:
            minimum = float(min_similarity)
        except (TypeError, ValueError, OverflowError):
            return original_assign(
                similarity_matrix,
                min_similarity=min_similarity,
                num_dummy=num_dummy,
                unmatched_value=unmatched_value,
                return_result=return_result,
            )
        if not math.isfinite(minimum):
            return original_assign(
                similarity_matrix,
                min_similarity=min_similarity,
                num_dummy=num_dummy,
                unmatched_value=unmatched_value,
                return_result=return_result,
            )

        finite_mask = roi_assignment_module.isfinite(similarities)
        if not bool(roi_assignment_module.any(finite_mask)):
            return original_assign(
                similarity_matrix,
                min_similarity=min_similarity,
                num_dummy=num_dummy,
                unmatched_value=unmatched_value,
                return_result=return_result,
            )

        maximum = float(roi_assignment_module.amax(similarities[finite_mask]))
        if _cost_conversion_is_finite(maximum, minimum):
            return original_assign(
                similarity_matrix,
                min_similarity=min_similarity,
                num_dummy=num_dummy,
                unmatched_value=unmatched_value,
                return_result=return_result,
            )

        assignment = _stable_extreme_assignment(
            roi_assignment_module,
            similarities,
            minimum=minimum,
            num_dummy=num_dummy,
            unmatched_value=unmatched_value,
        )
        if return_result:
            return roi_assignment_module._assignment_to_result(
                assignment,
                similarities,
                unmatched_value=unmatched_value,
            )
        return assignment

    assign_by_similarity_matrix.__name__ = getattr(
        original_assign,
        "__name__",
        "assign_by_similarity_matrix",
    )
    assign_by_similarity_matrix.__doc__ = getattr(original_assign, "__doc__", None)
    assign_by_similarity_matrix._pyrecest_finite_similarity_cost_range = True

    if hasattr(roi_assignment_module, storage_name):
        setattr(roi_assignment_module, storage_name, assign_by_similarity_matrix)
    else:
        roi_assignment_module.assign_by_similarity_matrix = assign_by_similarity_matrix
