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
        sys.float_info.epsilon
        * max(1.0, abs(max_similarity), abs(min_similarity)),
    )
    return math.isfinite(threshold_cost) and math.isfinite(
        threshold_cost + dummy_penalty
    )


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

        scale = max(1.0, abs(maximum), abs(minimum))
        normalized_similarities = similarities / scale
        assignment = original_assign(
            normalized_similarities,
            min_similarity=minimum / scale,
            num_dummy=num_dummy,
            unmatched_value=unmatched_value,
            return_result=False,
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
