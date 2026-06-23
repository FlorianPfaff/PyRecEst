"""Track-matrix edit what-if scoring utilities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .track_evaluation import normalize_track_matrix

TrackEditKind = Literal[
    "add_link", "remove_link", "swap_link", "split_track", "merge_tracks"
]
TrackLink = tuple[int, int, int, int]
_COUNT_KEYS = (
    "pairwise_true_positives",
    "pairwise_false_positives",
    "pairwise_false_negatives",
    "complete_track_true_positives",
    "complete_track_false_positives",
    "complete_track_false_negatives",
)


@dataclass(frozen=True)
class TrackEdit:
    """A local structural edit to a multi-session track matrix.

    The common link-edit fields describe an adjacent or non-adjacent link
    ``session_a:source_observation -> session_b:target_observation``. Extra
    edit-specific details, such as the wrong edge removed by a swap, live in
    ``metadata`` so domain packages can annotate edits without extending the
    generic API.
    """

    kind: TrackEditKind
    session_a: int | None = None
    session_b: int | None = None
    source_observation: int | None = None
    target_observation: int | None = None
    track_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackEditApplication:
    """Result of applying one structural edit to a track matrix."""

    edit: TrackEdit
    track_matrix: np.ndarray
    applied: bool
    action: str
    reason: str
    creates_duplicate_source: bool = False
    creates_duplicate_target: bool = False


@dataclass(frozen=True)
class TrackEditDelta:
    """Metric delta induced by a structural track edit."""

    edit: TrackEdit
    pairwise_tp_delta: int
    pairwise_fp_delta: int
    pairwise_fn_delta: int
    complete_tp_delta: int
    complete_fp_delta: int
    complete_fn_delta: int
    new_pairwise_f1: float
    new_complete_track_f1: float
    creates_duplicate_source: bool
    creates_duplicate_target: bool
    breaks_complete_track: bool
    applied: bool = False
    action: str = ""
    reason: str = ""


def apply_track_edit(track_matrix: Any, edit: TrackEdit) -> TrackEditApplication:
    """Apply ``edit`` to ``track_matrix`` and return the edited matrix.

    Missing observations are represented as ``None`` in the returned matrix.
    Callers that use integer ``-1`` sentinels can convert the result after the
    edit, while scoring functions in this module can consume it directly.
    """

    matrix = normalize_track_matrix(track_matrix)
    if edit.kind == "add_link":
        return _apply_add_link(matrix, edit)
    if edit.kind == "remove_link":
        return _apply_remove_link(matrix, edit)
    if edit.kind == "swap_link":
        return _apply_swap_link(matrix, edit)
    if edit.kind == "split_track":
        return _apply_split_track(matrix, edit)
    if edit.kind == "merge_tracks":
        return _apply_merge_tracks(matrix, edit)
    raise ValueError(f"Unsupported track edit kind: {edit.kind!r}")


def score_track_edit_delta(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    edit: TrackEdit,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
    complete_session_indices: Sequence[int] | None = None,
    count_duplicates: bool = False,
) -> TrackEditDelta:
    """Score the metric delta produced by one edit.

    By default, pairwise and complete tracks are scored as identity sets, matching
    PyRecEst's generic track-evaluation helpers.  Set ``count_duplicates=True``
    for benchmark protocols where duplicate predicted rows or links should count
    as false positives.
    """

    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)
    application = apply_track_edit(predicted, edit)
    baseline = _score_track_matrices(
        predicted,
        reference,
        session_pairs=session_pairs,
        complete_session_indices=complete_session_indices,
        count_duplicates=count_duplicates,
    )
    candidate = _score_track_matrices(
        application.track_matrix,
        reference,
        session_pairs=session_pairs,
        complete_session_indices=complete_session_indices,
        count_duplicates=count_duplicates,
    )
    deltas = {key: int(candidate[key]) - int(baseline[key]) for key in _COUNT_KEYS}
    return TrackEditDelta(
        edit=edit,
        pairwise_tp_delta=deltas["pairwise_true_positives"],
        pairwise_fp_delta=deltas["pairwise_false_positives"],
        pairwise_fn_delta=deltas["pairwise_false_negatives"],
        complete_tp_delta=deltas["complete_track_true_positives"],
        complete_fp_delta=deltas["complete_track_false_positives"],
        complete_fn_delta=deltas["complete_track_false_negatives"],
        new_pairwise_f1=float(candidate["pairwise_f1"]),
        new_complete_track_f1=float(candidate["complete_track_f1"]),
        creates_duplicate_source=bool(application.creates_duplicate_source),
        creates_duplicate_target=bool(application.creates_duplicate_target),
        breaks_complete_track=bool(
            int(candidate["complete_track_true_positives"])
            < int(baseline["complete_track_true_positives"])
        ),
        applied=bool(application.applied),
        action=str(application.action),
        reason=str(application.reason),
    )


def score_track_edits(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    edits: Iterable[TrackEdit],
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
    complete_session_indices: Sequence[int] | None = None,
    count_duplicates: bool = False,
) -> tuple[TrackEditDelta, ...]:
    """Score a collection of independent one-edit what-ifs."""

    edit_rows = tuple(edits)
    return tuple(
        score_track_edit_delta(
            predicted_track_matrix,
            reference_track_matrix,
            edit,
            session_pairs=session_pairs,
            complete_session_indices=complete_session_indices,
            count_duplicates=count_duplicates,
        )
        for edit in edit_rows
    )


def rank_track_edits_by_delta(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    edits: Iterable[TrackEdit],
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
    complete_session_indices: Sequence[int] | None = None,
    count_duplicates: bool = False,
) -> tuple[TrackEditDelta, ...]:
    """Return independent edits sorted by complete-track then pairwise F1 gain."""

    deltas = score_track_edits(
        predicted_track_matrix,
        reference_track_matrix,
        edits,
        session_pairs=session_pairs,
        complete_session_indices=complete_session_indices,
        count_duplicates=count_duplicates,
    )
    return tuple(sorted(deltas, key=_rank_key))


def _apply_add_link(matrix: np.ndarray, edit: TrackEdit) -> TrackEditApplication:
    session_a, session_b, source, target = _require_link_fields(edit)
    output = matrix.copy()
    source_rows = tuple(
        int(row) for row in np.flatnonzero(output[:, session_a] == source)
    )
    target_rows = tuple(
        int(row) for row in np.flatnonzero(output[:, session_b] == target)
    )
    duplicate_source = any(
        output[row_index, session_b] is not None
        and output[row_index, session_b] != target
        for row_index in source_rows
    )
    duplicate_target = any(
        output[row_index, session_a] is not None
        and output[row_index, session_a] != source
        for row_index in target_rows
    )
    if duplicate_source or duplicate_target:
        return TrackEditApplication(
            edit,
            output,
            False,
            "reject",
            "duplicate_source_or_target",
            creates_duplicate_source=bool(duplicate_source),
            creates_duplicate_target=bool(duplicate_target),
        )
    if len(source_rows) == 1 and len(target_rows) == 0:
        output[int(source_rows[0]), session_b] = int(target)
        return TrackEditApplication(edit, output, True, "insert_target", "accepted")
    if len(source_rows) == 0 and len(target_rows) == 1:
        output[int(target_rows[0]), session_a] = int(source)
        return TrackEditApplication(edit, output, True, "insert_source", "accepted")
    if len(source_rows) == 1 and len(target_rows) == 1:
        source_row = int(source_rows[0])
        target_row = int(target_rows[0])
        if source_row == target_row:
            return TrackEditApplication(
                edit, output, True, "already_same_component", "accepted"
            )
        merged = _merge_rows_if_compatible(output[source_row], output[target_row])
        if merged is not None:
            keep = [index for index in range(output.shape[0]) if index != target_row]
            output[source_row] = merged
            return TrackEditApplication(
                edit,
                output[np.asarray(keep, dtype=int)],
                True,
                "merge_components",
                "accepted",
            )
    if len(source_rows) == 0 and len(target_rows) == 0:
        new_row = np.empty((1, output.shape[1]), dtype=object)
        new_row[:] = None
        new_row[0, session_a] = int(source)
        new_row[0, session_b] = int(target)
        return TrackEditApplication(
            edit,
            np.vstack([output, new_row]),
            True,
            "add_partial_component",
            "accepted",
        )
    return TrackEditApplication(
        edit, output, False, "reject", "ambiguous_multiple_components"
    )


def _apply_remove_link(matrix: np.ndarray, edit: TrackEdit) -> TrackEditApplication:
    session_a, session_b, source, target = _require_link_fields(edit)
    occurrence_index = int(edit.metadata.get("occurrence_index", 0))
    output = matrix.copy()
    matching_rows = tuple(
        int(row)
        for row in np.flatnonzero(
            (output[:, session_a] == source) & (output[:, session_b] == target)
        )
    )
    if occurrence_index < 0 or occurrence_index >= len(matching_rows):
        return TrackEditApplication(edit, output, False, "reject", "edge_not_found")
    row_index = matching_rows[occurrence_index]
    left = output[row_index].copy()
    right = output[row_index].copy()
    left[session_b:] = None
    right[:session_b] = None
    pieces = [row for index, row in enumerate(output) if index != row_index]
    if any(value is not None for value in left):
        pieces.append(left)
    if any(value is not None for value in right):
        pieces.append(right)
    candidate = np.vstack(pieces).astype(object, copy=False) if pieces else output[:0]
    return TrackEditApplication(
        edit, candidate, True, "split_component_at_edge", "accepted"
    )


def _apply_swap_link(matrix: np.ndarray, edit: TrackEdit) -> TrackEditApplication:
    session_a, session_b, source, target = _require_link_fields(edit)
    wrong_session_a = int(edit.metadata.get("remove_session_a", session_a))
    wrong_session_b = int(edit.metadata.get("remove_session_b", session_b))
    wrong_source = _metadata_int(edit, "remove_source_observation")
    wrong_target = _metadata_int(edit, "remove_target_observation")
    output = matrix.copy()
    if (session_a, session_b) != (wrong_session_a, wrong_session_b):
        return TrackEditApplication(
            edit, output, False, "reject", "session_pair_mismatch"
        )
    matching_rows = tuple(
        int(row)
        for row in np.flatnonzero(
            (output[:, session_a] == wrong_source)
            & (output[:, session_b] == wrong_target)
        )
    )
    if not matching_rows:
        return TrackEditApplication(
            edit, output, False, "reject", "wrong_edge_not_found"
        )
    row_index = int(matching_rows[0])
    row_indices = np.arange(output.shape[0])
    duplicate_source = bool(
        source != wrong_source
        and np.any((output[:, session_a] == source) & (row_indices != row_index))
    )
    duplicate_target = bool(
        target != wrong_target
        and np.any((output[:, session_b] == target) & (row_indices != row_index))
    )
    output[row_index, session_a] = int(source)
    output[row_index, session_b] = int(target)
    return TrackEditApplication(
        edit,
        output,
        True,
        "swap_adjacent_edge",
        "accepted",
        creates_duplicate_source=duplicate_source,
        creates_duplicate_target=duplicate_target,
    )


def _apply_split_track(matrix: np.ndarray, edit: TrackEdit) -> TrackEditApplication:
    if edit.track_index is None:
        raise ValueError("split_track edits require track_index")
    if edit.session_b is None:
        raise ValueError("split_track edits require session_b as split point")
    row_index = int(edit.track_index)
    split_session = int(edit.session_b)
    output = matrix.copy()
    if row_index < 0 or row_index >= output.shape[0]:
        return TrackEditApplication(
            edit, output, False, "reject", "track_index_out_of_bounds"
        )
    if split_session <= 0 or split_session >= output.shape[1]:
        return TrackEditApplication(
            edit, output, False, "reject", "split_session_out_of_bounds"
        )
    left = output[row_index].copy()
    right = output[row_index].copy()
    left[split_session:] = None
    right[:split_session] = None
    pieces = [row for index, row in enumerate(output) if index != row_index]
    if any(value is not None for value in left):
        pieces.append(left)
    if any(value is not None for value in right):
        pieces.append(right)
    return TrackEditApplication(
        edit,
        np.vstack(pieces).astype(object, copy=False),
        True,
        "split_track",
        "accepted",
    )


def _apply_merge_tracks(matrix: np.ndarray, edit: TrackEdit) -> TrackEditApplication:
    if edit.track_index is None:
        raise ValueError("merge_tracks edits require track_index")
    other = edit.metadata.get("other_track_index")
    if other is None:
        raise ValueError("merge_tracks edits require metadata['other_track_index']")
    left_index = int(edit.track_index)
    right_index = int(other)
    output = matrix.copy()
    if (
        left_index < 0
        or right_index < 0
        or left_index >= output.shape[0]
        or right_index >= output.shape[0]
    ):
        return TrackEditApplication(
            edit, output, False, "reject", "track_index_out_of_bounds"
        )
    if left_index == right_index:
        return TrackEditApplication(
            edit, output, True, "already_same_component", "accepted"
        )
    merged = _merge_rows_if_compatible(output[left_index], output[right_index])
    if merged is None:
        return TrackEditApplication(edit, output, False, "reject", "row_conflict")
    keep = [index for index in range(output.shape[0]) if index != right_index]
    output[left_index] = merged
    return TrackEditApplication(
        edit, output[np.asarray(keep, dtype=int)], True, "merge_tracks", "accepted"
    )


def _require_link_fields(edit: TrackEdit) -> TrackLink:
    fields = (
        edit.session_a,
        edit.session_b,
        edit.source_observation,
        edit.target_observation,
    )
    if any(value is None for value in fields):
        raise ValueError(
            f"{edit.kind} edits require session_a, session_b, source_observation, and target_observation"
        )
    session_a, session_b, source, target = (
        int(value) for value in fields if value is not None
    )
    if session_a < 0 or session_b < 0 or session_a >= session_b:
        raise ValueError("edit sessions must satisfy 0 <= session_a < session_b")
    return session_a, session_b, source, target


def _metadata_int(edit: TrackEdit, name: str) -> int:
    if name not in edit.metadata:
        raise ValueError(f"swap_link edits require metadata[{name!r}]")
    return int(edit.metadata[name])


def _merge_rows_if_compatible(left: np.ndarray, right: np.ndarray) -> np.ndarray | None:
    values: list[int | None] = []
    for left_value, right_value in zip(left, right, strict=True):
        if (
            left_value is not None
            and right_value is not None
            and left_value != right_value
        ):
            return None
        values.append(left_value if left_value is not None else right_value)
    return np.asarray(values, dtype=object)


def _score_track_matrices(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    session_pairs: Iterable[tuple[int, int]] | None,
    complete_session_indices: Sequence[int] | None,
    count_duplicates: bool,
) -> dict[str, float | int]:
    if count_duplicates:
        pairwise = _score_multiset_identities(
            _track_link_counter(predicted, session_pairs=session_pairs),
            _track_link_counter(reference, session_pairs=session_pairs),
            prefix="pairwise",
        )
        complete = _score_multiset_identities(
            _complete_track_counter(
                predicted, session_indices=complete_session_indices
            ),
            _complete_track_counter(
                reference, session_indices=complete_session_indices
            ),
            prefix="complete_track",
        )
    else:
        pairwise = _score_set_identities(
            set(_track_link_counter(predicted, session_pairs=session_pairs)),
            set(_track_link_counter(reference, session_pairs=session_pairs)),
            prefix="pairwise",
        )
        complete = _score_set_identities(
            set(
                _complete_track_counter(
                    predicted, session_indices=complete_session_indices
                )
            ),
            set(
                _complete_track_counter(
                    reference, session_indices=complete_session_indices
                )
            ),
            prefix="complete_track",
        )
    return {**pairwise, **complete}


def _score_set_identities(
    predicted: set[Any], reference: set[Any], *, prefix: str
) -> dict[str, float | int]:
    true_positives = len(predicted & reference)
    false_positives = len(predicted - reference)
    false_negatives = len(reference - predicted)
    return _count_row(prefix, true_positives, false_positives, false_negatives)


def _score_multiset_identities(
    predicted: Counter[Any], reference: Counter[Any], *, prefix: str
) -> dict[str, float | int]:
    true_positives = int(sum((predicted & reference).values()))
    false_positives = int(sum(predicted.values())) - true_positives
    false_negatives = int(sum(reference.values())) - true_positives
    return _count_row(prefix, true_positives, false_positives, false_negatives)


def _count_row(
    prefix: str, true_positives: int, false_positives: int, false_negatives: int
) -> dict[str, float | int]:
    return {
        f"{prefix}_true_positives": int(true_positives),
        f"{prefix}_false_positives": int(false_positives),
        f"{prefix}_false_negatives": int(false_negatives),
        f"{prefix}_f1": _f1(true_positives, false_positives, false_negatives),
    }


def _track_link_counter(
    track_matrix: np.ndarray,
    *,
    session_pairs: Iterable[tuple[int, int]] | None,
) -> Counter[TrackLink]:
    counter: Counter[TrackLink] = Counter()
    for session_a, session_b in _session_pairs(track_matrix, session_pairs):
        for row in track_matrix:
            observation_a = row[session_a]
            observation_b = row[session_b]
            if observation_a is not None and observation_b is not None:
                counter[
                    (
                        int(session_a),
                        int(session_b),
                        int(observation_a),
                        int(observation_b),
                    )
                ] += 1
    return counter


def _complete_track_counter(
    track_matrix: np.ndarray,
    *,
    session_indices: Sequence[int] | None,
) -> Counter[tuple[int, ...]]:
    selected = (
        tuple(range(track_matrix.shape[1]))
        if session_indices is None
        else tuple(int(index) for index in session_indices)
    )
    counter: Counter[tuple[int, ...]] = Counter()
    for row in track_matrix:
        observations: list[int] = []
        for session_index in selected:
            value = row[int(session_index)]
            if value is None:
                break
            observations.append(int(value))
        else:
            counter[tuple(observations)] += 1
    return counter


def _session_pairs(
    matrix: np.ndarray,
    session_pairs: Iterable[tuple[int, int]] | None,
) -> tuple[tuple[int, int], ...]:
    if session_pairs is None:
        return tuple((index, index + 1) for index in range(max(0, matrix.shape[1] - 1)))
    pairs = tuple(
        (int(session_a), int(session_b)) for session_a, session_b in session_pairs
    )
    for session_a, session_b in pairs:
        if session_a < 0 or session_b <= session_a:
            raise ValueError("session pairs must satisfy 0 <= session_a < session_b")
        if session_b >= matrix.shape[1]:
            raise IndexError(
                f"session pair {(session_a, session_b)} out of bounds for {matrix.shape[1]} sessions"
            )
    return pairs


def _validate_compatible_shapes(predicted: np.ndarray, reference: np.ndarray) -> None:
    if predicted.ndim != 2 or reference.ndim != 2:
        raise ValueError("track matrices must have shape (n_tracks, n_sessions)")
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError(
            "predicted and reference track matrices must have the same number of sessions"
        )


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * int(tp) + int(fp) + int(fn)
    if denominator == 0:
        return 1.0
    return float(2 * int(tp) / denominator)


def _rank_key(delta: TrackEditDelta) -> tuple[float, float, bool, bool, bool, str]:
    return (
        -float(delta.new_complete_track_f1),
        -float(delta.new_pairwise_f1),
        not delta.applied,
        bool(delta.creates_duplicate_source),
        bool(delta.creates_duplicate_target),
        str(delta.edit.kind),
    )
