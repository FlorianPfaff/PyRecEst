"""Score-native conveniences for multi-session association."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from pyrecest.backend import (
    __backend_name__,
)
from pyrecest.backend import all as backend_all  # pylint: disable=no-name-in-module
from pyrecest.backend import (
    asarray,
)
from pyrecest.backend import copy as backend_copy
from pyrecest.backend import (
    full,
    isfinite,
    where,
)

from .multisession_assignment import (
    MultiSessionAssignmentResult,
    PairwiseCostsInput,
    SessionSizesInput,
    TrackInput,
    _infer_and_validate_session_sizes,
    _iter_track_items,
    _normalize_pairwise_costs,
    _normalize_session_sizes,
    _validate_track_session_sizes,
    solve_multisession_assignment,
)

_INVALID_SCORE_SCALAR_TYPES = (
    type(None),
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
    complex,
    np.complexfloating,
    np.datetime64,
    np.timedelta64,
)
_REJECTED_SCORE_ARRAY_KINDS = frozenset({"b", "c", "S", "U", "M", "m"})


def _ensure_supported_backend(feature_name: str) -> None:
    if __backend_name__ == "jax":
        raise NotImplementedError(
            f"{feature_name} is not supported on the JAX backend."
        )


def _default_score_to_cost(scores: Any) -> Any:
    return -asarray(scores, dtype=float)


def _is_text_scalar(value: Any) -> bool:
    return isinstance(value, (str, bytes, np.str_, np.bytes_))


def _raise_invalid_score_matrix() -> None:
    raise ValueError("pairwise_scores must contain real numeric score matrices.")


def _validate_real_score_matrix(value: Any) -> None:
    try:
        raw_values = np.asarray(value)
    except (TypeError, ValueError, RuntimeError):
        return

    if raw_values.dtype.kind in _REJECTED_SCORE_ARRAY_KINDS:
        _raise_invalid_score_matrix()
    if raw_values.dtype == object:
        for item in raw_values.reshape(-1):
            if isinstance(item, _INVALID_SCORE_SCALAR_TYPES):
                _raise_invalid_score_matrix()


def _validate_pairwise_score_inputs(pairwise_scores: PairwiseCostsInput) -> None:
    if isinstance(pairwise_scores, Mapping):
        score_matrices = pairwise_scores.values()
    else:
        score_matrices = pairwise_scores

    for score_matrix in score_matrices:
        _validate_real_score_matrix(score_matrix)


def _normalize_min_score(min_score: Any) -> float:
    min_score_array = np.asarray(min_score)
    if min_score_array.shape != () or min_score_array.dtype == np.bool_:
        raise ValueError("min_score must be a finite scalar.")

    min_score_value = min_score_array.item()
    if isinstance(min_score_value, (bool, np.bool_)) or _is_text_scalar(
        min_score_value
    ):
        raise ValueError("min_score must be a finite scalar.")

    try:
        min_score_float = float(min_score_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("min_score must be a finite scalar.") from exc
    if not math.isfinite(min_score_float):
        raise ValueError("min_score must be a finite scalar.")
    return min_score_float


def _normalize_max_gap(max_gap: Any) -> int:
    max_gap_array = np.asarray(max_gap)
    if max_gap_array.shape != () or max_gap_array.dtype == np.bool_:
        raise ValueError("max_gap must be a non-negative integer.")

    max_gap_value = max_gap_array.item()
    if isinstance(max_gap_value, (bool, np.bool_)) or _is_text_scalar(max_gap_value):
        raise ValueError("max_gap must be a non-negative integer.")

    if isinstance(max_gap_value, (int, np.integer)):
        if max_gap_value < 0:
            raise ValueError("max_gap must be a non-negative integer.")
        return int(max_gap_value)

    try:
        max_gap_float = float(max_gap_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_gap must be a non-negative integer.") from exc
    if (
        not math.isfinite(max_gap_float)
        or max_gap_float < 0
        or not max_gap_float.is_integer()
    ):
        raise ValueError("max_gap must be a non-negative integer.")
    return int(max_gap_float)


def _normalize_index_matrix_fill_value(fill_value: Any) -> int:
    fill_value_array = np.asarray(fill_value)
    if fill_value_array.shape != () or fill_value_array.dtype == np.bool_:
        raise ValueError("fill_value must be a negative integer.")

    fill_value_value = fill_value_array.item()
    if isinstance(fill_value_value, (bool, np.bool_)) or _is_text_scalar(
        fill_value_value
    ):
        raise ValueError("fill_value must be a negative integer.")

    if isinstance(fill_value_value, (int, np.integer)):
        integer_fill_value = int(fill_value_value)
    else:
        try:
            fill_value_float = float(fill_value_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("fill_value must be a negative integer.") from exc
        if not math.isfinite(fill_value_float) or not fill_value_float.is_integer():
            raise ValueError("fill_value must be a negative integer.")
        integer_fill_value = int(fill_value_float)

    if integer_fill_value >= 0:
        raise ValueError("fill_value must be a negative integer.")
    return integer_fill_value


def tracks_to_index_matrix(
    tracks: list[TrackInput],
    session_sizes: SessionSizesInput | None = None,
    *,
    fill_value: int = -1,
):
    """Convert tracks to a dense ``track x session`` ROI-index matrix."""
    _ensure_supported_backend("tracks_to_index_matrix")
    fill_value = _normalize_index_matrix_fill_value(fill_value)

    _, max_session_index = _validate_track_session_sizes(
        tracks,
        session_sizes,
        require_unique_sessions=True,
    )

    matrix = full((len(tracks), max_session_index + 1), fill_value, dtype=int)
    seen_observations: set[tuple[int, int]] = set()
    for track_index, track in enumerate(tracks):
        for session_index, detection_index in _iter_track_items(track):
            observation = (int(session_index), int(detection_index))
            if observation in seen_observations:
                raise ValueError("Each detection can only belong to a single track.")
            seen_observations.add(observation)
            matrix[track_index, session_index] = detection_index
    return matrix


def solve_multisession_assignment_from_similarity(  # pylint: disable=R0913,R0914
    pairwise_scores: PairwiseCostsInput,
    session_sizes: SessionSizesInput | None = None,
    *,
    min_score: float | None = None,
    max_gap: int | None = None,
    gap_penalty: float = 0.0,
    start_cost: float = 0.0,
    end_cost: float = 0.0,
    score_to_cost: Callable[[Any], Any] | None = None,
) -> MultiSessionAssignmentResult:
    """Score-native wrapper around :func:`solve_multisession_assignment`."""
    _ensure_supported_backend("solve_multisession_assignment_from_similarity")

    if min_score is not None:
        min_score = _normalize_min_score(min_score)
    if max_gap is not None:
        max_gap = _normalize_max_gap(max_gap)

    _validate_pairwise_score_inputs(pairwise_scores)
    normalized_pairwise_scores = _normalize_pairwise_costs(pairwise_scores)
    normalized_session_sizes = _normalize_session_sizes(session_sizes)
    session_sizes_map = _infer_and_validate_session_sizes(
        normalized_pairwise_scores,
        normalized_session_sizes,
    )
    if score_to_cost is None:
        score_to_cost = _default_score_to_cost

    transformed_pairwise_costs: dict[tuple[int, int], Any] = {}
    for (
        source_session,
        target_session,
    ), score_matrix in normalized_pairwise_scores.items():
        gap = int(target_session) - int(source_session) - 1
        if max_gap is not None and gap > max_gap:
            continue

        score_matrix_array = asarray(score_matrix, dtype=float)
        score_finite_mask = isfinite(score_matrix_array)
        admissible_mask = backend_copy(score_finite_mask)
        if min_score is not None:
            admissible_mask &= score_matrix_array >= min_score

        safe_scores = where(score_finite_mask, score_matrix_array, 0.0)
        cost_matrix = asarray(score_to_cost(safe_scores), dtype=float)
        if cost_matrix.shape != score_matrix_array.shape:
            raise ValueError("score_to_cost must preserve the input matrix shape.")

        cost_matrix = backend_copy(cost_matrix)
        cost_matrix[~admissible_mask] = math.inf
        if not bool(backend_all(isfinite(cost_matrix[admissible_mask]))):
            raise ValueError(
                "score_to_cost must return finite costs for admissible scores."
            )
        transformed_pairwise_costs[(source_session, target_session)] = cost_matrix

    return solve_multisession_assignment(
        transformed_pairwise_costs,
        session_sizes=session_sizes_map,
        start_cost=start_cost,
        end_cost=end_cost,
        gap_penalty=gap_penalty,
    )


def stitch_tracks_from_pairwise_scores(
    pairwise_scores: PairwiseCostsInput,
    session_sizes: SessionSizesInput | None = None,
    **kwargs,
) -> MultiSessionAssignmentResult:
    """Track2p-style alias for the score-native wrapper."""
    return solve_multisession_assignment_from_similarity(
        pairwise_scores,
        session_sizes=session_sizes,
        **kwargs,
    )


def _result_to_index_matrix(
    self, session_sizes: SessionSizesInput | None = None, *, fill_value: int = -1
):
    return tracks_to_index_matrix(
        self.tracks, session_sizes=session_sizes, fill_value=fill_value
    )


MultiSessionAssignmentResult.to_index_matrix = _result_to_index_matrix  # type: ignore[attr-defined]
