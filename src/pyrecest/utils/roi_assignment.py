"""Utilities for overlap-aware ROI association.

These helpers are designed for session-to-session identity matching problems where
binary regions of interest (ROIs) are the primary observation. They are especially
useful for calcium-imaging pipelines that represent segmented cells either as dense
boolean masks or as sparse pixel lists such as Suite2p ``stat.npy`` entries with
``ypix``/``xpix`` coordinates.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    amax,
    amin,
    any,
    argmax,
    array,
    asarray,
    cumsum,
    flip,
    float64,
    full,
    full_like,
    int64,
    isfinite,
    linalg,
    linspace,
    mean,
    nonzero,
    where,
    zeros,
)
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


@dataclass(frozen=True, slots=True)
class SimilarityAssignmentResult:
    """Result of solving a similarity-based one-to-one assignment problem."""

    assignment: Any
    matched_row_indices: Any
    matched_col_indices: Any
    matched_similarities: Any
    unmatched_row_indices: Any
    unmatched_col_indices: Any

    def as_row_to_col_map(self) -> dict[int, int]:
        """Return accepted row-to-column matches as a mapping."""

        return {
            int(row_index): int(col_index)
            for row_index, col_index in zip(
                self.matched_row_indices.tolist(),
                self.matched_col_indices.tolist(),
            )
        }


@dataclass(frozen=True, slots=True)
class ROIAssociationResult:  # pylint: disable=too-many-instance-attributes
    """Container holding the outcome of ROI association."""

    assignment: Any
    similarity_matrix: Any
    matched_reference_indices: Any
    matched_query_indices: Any
    matched_similarities: Any
    unmatched_reference_indices: Any
    unmatched_query_indices: Any
    acceptance_threshold: float | None
    threshold_method: str | None
    centroid_distance_matrix: Any | None = None

    def as_reference_to_query_map(self) -> dict[int, int]:
        """Return accepted reference-to-query matches as a mapping."""

        return {
            int(reference_index): int(query_index)
            for reference_index, query_index in zip(
                self.matched_reference_indices.tolist(),
                self.matched_query_indices.tolist(),
            )
        }


@dataclass(frozen=True, slots=True)
class _PreparedROI:
    pixels: set[tuple[int, int]]
    centroid: Any
    area: int


def _backend_not_supported(function_name: str) -> None:
    if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
        raise NotImplementedError(
            f"{function_name} is not supported on the jax backend."
        )


def _to_backend_array(array_like, *, dtype=None):
    if hasattr(array_like, "tolist"):
        return asarray(array_like.tolist(), dtype=dtype)
    return asarray(array_like, dtype=dtype)


def _as_int_set(indices) -> set[int]:
    values = indices.tolist() if hasattr(indices, "tolist") else indices
    return {int(index) for index in values}


def _setdiff_indices(size: int, selected_indices):
    selected = _as_int_set(selected_indices)
    return array([index for index in range(size) if index not in selected], dtype=int64)


def _histogram(values, *, bins: int, value_min: float, value_max: float):
    if bins <= 0:
        raise ValueError("nbins must be positive.")

    bin_edges = linspace(value_min, value_max, bins + 1)
    scale = bins / (value_max - value_min)
    counts = [0.0] * bins
    value_list = values.tolist() if hasattr(values, "tolist") else values
    for value in value_list:
        bin_index = int((float(value) - value_min) * scale)
        if bin_index == bins:
            bin_index = bins - 1
        if 0 <= bin_index < bins:
            counts[bin_index] += 1.0
    return array(counts, dtype=float64), bin_edges


def _extract_roi_support(roi) -> set[tuple[int, int]]:
    """Return the binary support of an ROI as a set of ``(y, x)`` pixel tuples.

    Supported ROI formats are:

    * 2D dense arrays, where all non-zero entries are treated as active pixels.
    * ``(ypix, xpix)`` tuples.
    * mappings with ``"ypix"`` and ``"xpix"`` keys, e.g. Suite2p ``stat`` entries.

    Lists are intentionally *not* treated as sparse ``(ypix, xpix)`` inputs because
    a dense mask can also be represented as a Python list with length two.
    """

    if isinstance(roi, Mapping):
        if "ypix" not in roi or "xpix" not in roi:
            raise KeyError(
                "Sparse ROI mappings must provide both 'ypix' and 'xpix' keys."
            )
        ypix = asarray(roi["ypix"], dtype=int64).ravel()
        xpix = asarray(roi["xpix"], dtype=int64).ravel()
    elif isinstance(roi, tuple) and len(roi) == 2:
        ypix = asarray(roi[0], dtype=int64).ravel()
        xpix = asarray(roi[1], dtype=int64).ravel()
    else:
        mask = asarray(roi)
        if mask.ndim != 2:
            raise ValueError(
                "Dense ROI representations must be two-dimensional arrays."
            )
        nonzero_result = nonzero(mask)
        if isinstance(nonzero_result, tuple):
            ypix, xpix = nonzero_result
        else:
            ypix = nonzero_result[:, 0]
            xpix = nonzero_result[:, 1]
        ypix = asarray(ypix, dtype=int64)
        xpix = asarray(xpix, dtype=int64)

    if ypix.shape != xpix.shape:
        raise ValueError(
            "ROI coordinate vectors 'ypix' and 'xpix' must have the same length."
        )

    return {
        (int(y_index), int(x_index))
        for y_index, x_index in zip(ypix.tolist(), xpix.tolist())
    }


def _prepare_roi(roi) -> _PreparedROI:
    pixels = _extract_roi_support(roi)
    area = len(pixels)
    if area == 0:
        centroid = array([float("nan"), float("nan")], dtype=float64)
    else:
        coords = asarray(tuple(pixels), dtype=float64)
        centroid = mean(coords, axis=0)
    return _PreparedROI(pixels=pixels, centroid=centroid, area=area)


def _prepare_rois(rois: Sequence) -> list[_PreparedROI]:
    return [_prepare_roi(roi) for roi in rois]


def _assignment_to_result(
    assignment,
    similarity_matrix,
    *,
    unmatched_value: int,
) -> SimilarityAssignmentResult:
    similarities = _to_backend_array(similarity_matrix, dtype=float64)
    assignment_array = _to_backend_array(assignment, dtype=int64)
    matched_row_indices = asarray(
        where(assignment_array != unmatched_value)[0], dtype=int64
    )
    matched_col_indices = asarray(assignment_array[matched_row_indices], dtype=int64)
    if matched_row_indices.shape[0] == 0:
        matched_similarities = zeros((0,), dtype=float64)
    else:
        matched_similarities = asarray(
            similarities[matched_row_indices, matched_col_indices], dtype=float64
        )
    unmatched_row_indices = asarray(
        where(assignment_array == unmatched_value)[0], dtype=int64
    )
    unmatched_col_indices = _setdiff_indices(
        similarities.shape[1],
        matched_col_indices,
    )
    return SimilarityAssignmentResult(
        assignment=assignment_array,
        matched_row_indices=matched_row_indices,
        matched_col_indices=matched_col_indices,
        matched_similarities=matched_similarities,
        unmatched_row_indices=unmatched_row_indices,
        unmatched_col_indices=unmatched_col_indices,
    )


def _result_to_assignment(
    result: SimilarityAssignmentResult,
    *,
    n_rows: int,
    unmatched_value: int,
):
    assignment = full((n_rows,), unmatched_value, dtype=int64)
    if result.matched_row_indices.shape[0] > 0:
        assignment[result.matched_row_indices] = result.matched_col_indices
    return assignment


def roi_iou(roi_a, roi_b) -> float:
    """Compute the intersection over union (IoU) between two ROIs.

    The ROIs may be provided either as dense masks or sparse pixel coordinates.
    Empty-empty pairs return ``0.0``.
    """

    prepared_a = _prepare_roi(roi_a)
    prepared_b = _prepare_roi(roi_b)

    if prepared_a.area == 0 and prepared_b.area == 0:
        return 0.0
    intersection = len(prepared_a.pixels & prepared_b.pixels)
    union = prepared_a.area + prepared_b.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def roi_centroid(roi):
    """Return the centroid of an ROI as ``[y, x]``."""

    return _to_backend_array(_prepare_roi(roi).centroid, dtype=float64)


def pairwise_centroid_distances(reference_rois: Sequence, query_rois: Sequence):
    """Return a dense matrix of pairwise ROI centroid distances."""

    _backend_not_supported("pairwise_centroid_distances")
    prepared_reference = _prepare_rois(reference_rois)
    prepared_query = _prepare_rois(query_rois)
    n_reference = len(prepared_reference)
    n_query = len(prepared_query)
    distances = full((n_reference, n_query), float("inf"), dtype=float64)

    for row_index, reference in enumerate(prepared_reference):
        for col_index, query in enumerate(prepared_query):
            if reference.area == 0 and query.area == 0:
                distances[row_index, col_index] = 0.0
                continue
            if reference.area == 0 or query.area == 0:
                continue
            distances[row_index, col_index] = float(
                linalg.norm(reference.centroid - query.centroid)
            )
    return distances


# pylint: disable=too-many-locals
def pairwise_iou_masks(
    reference_rois: Sequence,
    query_rois: Sequence,
    *,
    centroid_distance_threshold: float | None = None,
    return_centroid_distance_matrix: bool = False,
):
    """Compute a dense pairwise IoU matrix for two ROI collections."""

    _backend_not_supported("pairwise_iou_masks")
    if centroid_distance_threshold is not None and centroid_distance_threshold < 0:
        raise ValueError("centroid_distance_threshold must be non-negative.")

    prepared_reference = _prepare_rois(reference_rois)
    prepared_query = _prepare_rois(query_rois)
    n_reference = len(prepared_reference)
    n_query = len(prepared_query)
    iou_matrix = zeros((n_reference, n_query), dtype=float64)
    centroid_distance_matrix = full((n_reference, n_query), float("inf"), dtype=float64)

    if n_reference == 0 or n_query == 0:
        if return_centroid_distance_matrix:
            return iou_matrix, centroid_distance_matrix
        return iou_matrix

    for row_index, reference in enumerate(prepared_reference):
        for col_index, query in enumerate(prepared_query):
            if reference.area == 0 and query.area == 0:
                centroid_distance_matrix[row_index, col_index] = 0.0
                continue
            if reference.area == 0 or query.area == 0:
                continue

            centroid_distance = float(linalg.norm(reference.centroid - query.centroid))
            centroid_distance_matrix[row_index, col_index] = centroid_distance
            if (
                centroid_distance_threshold is not None
                and centroid_distance > centroid_distance_threshold
            ):
                continue

            intersection = len(reference.pixels & query.pixels)
            union = reference.area + query.area - intersection
            if union > 0:
                iou_matrix[row_index, col_index] = intersection / union

    if return_centroid_distance_matrix:
        return iou_matrix, centroid_distance_matrix
    return iou_matrix


# pylint: disable=too-many-locals
def otsu_similarity_threshold(similarities, *, nbins: int = 256) -> float:
    """Estimate a threshold using Otsu's method on one-dimensional similarities."""

    values = asarray(similarities, dtype=float64)
    values = values[isfinite(values)]
    if values.shape[0] == 0:
        return 0.0

    value_min = float(amin(values))
    value_max = float(amax(values))
    if value_min == value_max:
        return value_min

    histogram, bin_edges = _histogram(
        values,
        bins=nbins,
        value_min=value_min,
        value_max=value_max,
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    weight_background = cumsum(histogram)
    reversed_histogram = flip(histogram, axis=0)
    histogram_weighted_centers = histogram * bin_centers
    weight_foreground = flip(cumsum(reversed_histogram), axis=0)
    weighted_sum_background = cumsum(histogram_weighted_centers)
    weighted_sum_foreground = flip(
        cumsum(flip(histogram_weighted_centers, axis=0)),
        axis=0,
    )

    valid_mask = (weight_background > 0) & (weight_foreground > 0)
    if not bool(any(valid_mask)):
        return 0.0

    background_denominator = where(valid_mask, weight_background, 1.0)
    foreground_denominator = where(valid_mask, weight_foreground, 1.0)
    mean_background = where(
        valid_mask,
        weighted_sum_background / background_denominator,
        0.0,
    )
    mean_foreground = where(
        valid_mask,
        weighted_sum_foreground / foreground_denominator,
        0.0,
    )

    between_class_variance = where(
        valid_mask,
        weight_background
        * weight_foreground
        * (mean_background - mean_foreground) ** 2,
        0.0,
    )

    best_index = int(argmax(between_class_variance))
    return float(bin_centers[best_index])


# pylint: disable=too-many-branches,too-many-locals
def minimum_similarity_threshold(similarities, *, nbins: int = 256) -> float:
    """Estimate a threshold by locating a valley between the two strongest modes."""

    values = asarray(similarities, dtype=float64)
    values = values[isfinite(values)]
    if values.shape[0] == 0:
        return 0.0

    value_min = float(amin(values))
    value_max = float(amax(values))
    if value_min == value_max:
        return value_min

    histogram, bin_edges = _histogram(
        values,
        bins=nbins,
        value_min=value_min,
        value_max=value_max,
    )
    histogram_values = histogram.tolist() if hasattr(histogram, "tolist") else histogram
    peak_indices, _ = find_peaks(histogram_values)
    peak_indices_list = [int(index) for index in peak_indices.tolist()]

    if len(peak_indices_list) < 2:
        return otsu_similarity_threshold(values, nbins=nbins)

    best_peak_pair = None
    best_peak_score = None
    for left_position, left_peak in enumerate(peak_indices_list[:-1]):
        for right_peak in peak_indices_list[left_position + 1 :]:
            valley_min = min(histogram_values[left_peak : right_peak + 1])
            peak_score = (
                min(histogram_values[left_peak], histogram_values[right_peak])
                - valley_min,
                right_peak - left_peak,
                histogram_values[left_peak] + histogram_values[right_peak],
            )
            if best_peak_score is None or peak_score > best_peak_score:
                best_peak_pair = (left_peak, right_peak)
                best_peak_score = peak_score

    if best_peak_pair is None:
        return otsu_similarity_threshold(values, nbins=nbins)

    left_peak, right_peak = best_peak_pair
    if right_peak - left_peak <= 1:
        return otsu_similarity_threshold(values, nbins=nbins)

    valley_values = histogram_values[left_peak : right_peak + 1]
    valley_min = min(valley_values)
    best_plateau_start = -1
    best_plateau_end = -1
    plateau_start = -1
    plateau_end = -1
    for offset, value in enumerate(valley_values):
        if value == valley_min:
            if plateau_start < 0:
                plateau_start = left_peak + offset
            plateau_end = left_peak + offset
            continue
        if plateau_start >= 0 and (
            best_plateau_start < 0
            or plateau_end - plateau_start > best_plateau_end - best_plateau_start
        ):
            best_plateau_start = plateau_start
            best_plateau_end = plateau_end
        plateau_start = -1

    if plateau_start >= 0 and (
        best_plateau_start < 0
        or plateau_end - plateau_start > best_plateau_end - best_plateau_start
    ):
        best_plateau_start = plateau_start
        best_plateau_end = plateau_end

    valley_index = (best_plateau_start + best_plateau_end) // 2
    return float(0.5 * (bin_edges[valley_index] + bin_edges[valley_index + 1]))


# pylint: disable=too-many-branches,too-many-locals,too-many-return-statements
def assign_by_similarity_matrix(
    similarity_matrix,
    min_similarity: float = 0.0,
    num_dummy: int | None = None,
    unmatched_value: int = -1,
    *,
    return_result: bool = False,
):
    """Solve a one-to-one assignment problem by maximizing similarity."""

    _backend_not_supported("assign_by_similarity_matrix")
    similarities = asarray(similarity_matrix, dtype=float64)
    if similarities.ndim != 2:
        raise ValueError("similarity_matrix must be two-dimensional.")

    n_rows, n_cols = similarities.shape
    if n_rows == 0:
        empty_assignment = zeros((0,), dtype=int64)
        if return_result:
            return _assignment_to_result(
                empty_assignment,
                zeros((0, n_cols), dtype=float64),
                unmatched_value=unmatched_value,
            )
        return empty_assignment
    if n_cols == 0:
        assignment = full((n_rows,), unmatched_value, dtype=int64)
        if return_result:
            return _assignment_to_result(
                assignment,
                zeros((n_rows, 0), dtype=float64),
                unmatched_value=unmatched_value,
            )
        return assignment

    finite_mask = isfinite(similarities)
    if not bool(any(finite_mask)):
        assignment = full((n_rows,), unmatched_value, dtype=int64)
        if return_result:
            return _assignment_to_result(
                assignment,
                _to_backend_array(similarities, dtype=float64),
                unmatched_value=unmatched_value,
            )
        return assignment

    if num_dummy is None:
        num_dummy = max(n_rows, n_cols)
    if num_dummy < 0:
        raise ValueError("num_dummy must be non-negative.")

    max_similarity = float(amax(similarities[finite_mask]))
    threshold_cost = max_similarity - float(min_similarity)
    dummy_penalty = max(
        1e-12,
        sys.float_info.epsilon * max(1.0, abs(max_similarity), abs(min_similarity)),
    )
    dummy_cost = threshold_cost + dummy_penalty

    valid_mask = finite_mask & (similarities >= min_similarity)
    cost_matrix = full_like(similarities, dummy_cost)
    cost_matrix[valid_mask] = max_similarity - similarities[valid_mask]

    padded_size = max(n_rows, n_cols) + int(num_dummy)
    padded_cost = full((padded_size, padded_size), dummy_cost, dtype=float64)
    padded_cost[:n_rows, :n_cols] = cost_matrix

    row_ind, col_ind = linear_sum_assignment(padded_cost)

    assignment = full((n_rows,), unmatched_value, dtype=int64)
    for row_index, col_index in zip(row_ind, col_ind):
        if row_index >= n_rows:
            continue
        if col_index < n_cols and valid_mask[row_index, col_index]:
            assignment[row_index] = int(col_index)

    if return_result:
        return _assignment_to_result(
            assignment,
            _to_backend_array(similarities, dtype=float64),
            unmatched_value=unmatched_value,
        )
    return assignment


def _filter_matches_by_threshold(
    assignment_result: SimilarityAssignmentResult,
    *,
    threshold: float,
    n_rows: int,
    n_cols: int,
    unmatched_value: int,
) -> SimilarityAssignmentResult:
    keep_mask = assignment_result.matched_similarities >= float(threshold)
    filtered_matched_rows = assignment_result.matched_row_indices[keep_mask]
    filtered_matched_cols = assignment_result.matched_col_indices[keep_mask]
    filtered_matched_similarities = assignment_result.matched_similarities[keep_mask]

    filtered_assignment = full((n_rows,), unmatched_value, dtype=int64)
    if filtered_matched_rows.shape[0] > 0:
        filtered_assignment[filtered_matched_rows] = filtered_matched_cols

    unmatched_rows = _setdiff_indices(n_rows, filtered_matched_rows)
    unmatched_cols = _setdiff_indices(n_cols, filtered_matched_cols)

    return SimilarityAssignmentResult(
        assignment=filtered_assignment,
        matched_row_indices=filtered_matched_rows,
        matched_col_indices=filtered_matched_cols,
        matched_similarities=filtered_matched_similarities,
        unmatched_row_indices=unmatched_rows,
        unmatched_col_indices=unmatched_cols,
    )


# pylint: disable=too-many-arguments,too-many-branches,too-many-positional-arguments
def associate_rois_by_iou(
    reference_rois: Sequence,
    query_rois: Sequence,
    min_iou: float = 0.0,
    num_dummy: int | None = None,
    unmatched_value: int = -1,
    return_iou_matrix: bool = False,
    *,
    centroid_distance_threshold: float | None = None,
    threshold_method: str | None = None,
    require_positive_match: bool = True,
    return_result: bool = False,
):
    """Associate ROIs by maximizing global IoU under one-to-one constraints."""

    iou_matrix, centroid_distance_matrix = pairwise_iou_masks(
        reference_rois,
        query_rois,
        centroid_distance_threshold=centroid_distance_threshold,
        return_centroid_distance_matrix=True,
    )

    effective_min_iou = float(min_iou)
    if require_positive_match and effective_min_iou <= 0.0:
        effective_min_iou = sys.float_info.min

    assignment_result = assign_by_similarity_matrix(
        iou_matrix,
        min_similarity=effective_min_iou,
        num_dummy=num_dummy,
        unmatched_value=unmatched_value,
        return_result=True,
    )

    acceptance_threshold = None
    normalized_threshold_method = None
    if threshold_method is not None:
        normalized_threshold_method = threshold_method.lower()
        if normalized_threshold_method == "otsu":
            acceptance_threshold = otsu_similarity_threshold(
                assignment_result.matched_similarities
            )
        elif normalized_threshold_method == "min":
            acceptance_threshold = minimum_similarity_threshold(
                assignment_result.matched_similarities
            )
        else:
            raise ValueError("threshold_method must be one of None, 'otsu', or 'min'.")

        acceptance_threshold = float(max(acceptance_threshold, effective_min_iou))
        assignment_result = _filter_matches_by_threshold(
            assignment_result,
            threshold=acceptance_threshold,
            n_rows=len(reference_rois),
            n_cols=len(query_rois),
            unmatched_value=unmatched_value,
        )

    if return_result:
        result = ROIAssociationResult(
            assignment=asarray(assignment_result.assignment, dtype=int64),
            similarity_matrix=_to_backend_array(iou_matrix, dtype=float64),
            matched_reference_indices=assignment_result.matched_row_indices,
            matched_query_indices=assignment_result.matched_col_indices,
            matched_similarities=assignment_result.matched_similarities,
            unmatched_reference_indices=assignment_result.unmatched_row_indices,
            unmatched_query_indices=assignment_result.unmatched_col_indices,
            acceptance_threshold=acceptance_threshold,
            threshold_method=normalized_threshold_method,
            centroid_distance_matrix=centroid_distance_matrix,
        )
        if return_iou_matrix:
            return result, iou_matrix
        return result

    assignment = asarray(
        _result_to_assignment(
            assignment_result,
            n_rows=len(reference_rois),
            unmatched_value=unmatched_value,
        ),
        dtype=int64,
    )
    if return_iou_matrix:
        return assignment, iou_matrix
    return assignment
