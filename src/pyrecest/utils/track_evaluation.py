"""Generic evaluation helpers for multi-session track matrices."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias

import numpy as np

Observation: TypeAlias = tuple[int, int]
TrackLink: TypeAlias = tuple[int, int, int, int]

_MISSING = object()
_MISSING_STRINGS = {"", "none", "nan", "null"}

__all__ = (
    "Observation",
    "TrackLink",
    "complete_track_set",
    "normalize_track_matrix",
    "pairwise_track_set",
    "reference_fragment_counts",
    "score_complete_tracks",
    "score_false_continuations",
    "score_fragmentation",
    "score_pairwise_tracks",
    "score_track_fragmentation",
    "score_track_links",
    "score_track_matrices",
    "summarize_track_errors",
    "summarize_tracks",
    "track_error_ledger",
    "track_lengths",
    "track_pair_set",
)


def normalize_track_matrix(track_matrix: Any) -> np.ndarray:
    """Return an object matrix containing integer observation indices or ``None``."""
    matrix = np.asarray(track_matrix, dtype=object)
    if matrix.ndim != 2:
        raise ValueError("track_matrix must have shape (n_tracks, n_sessions)")
    normalized = np.empty(matrix.shape, dtype=object)
    for index, value in np.ndenumerate(matrix):
        normalized[index] = _parse_optional_int(value)
    return normalized


def track_lengths(track_matrix: Any) -> np.ndarray:
    """Return the number of present observations in each track row."""
    matrix = normalize_track_matrix(track_matrix)
    return np.asarray(
        [sum(value is not None for value in row) for row in matrix], dtype=int
    )


def complete_track_set(
    track_matrix: Any, *, session_indices: Sequence[int] | None = None
) -> set[tuple[int, ...]]:
    """Return exact full-track tuples present in every selected session."""
    matrix = normalize_track_matrix(track_matrix)
    complete_tracks: set[tuple[int, ...]] = set()
    for row in matrix:
        values = [
            row[session_idx]
            for session_idx in _selected_sessions(matrix, session_indices)
        ]
        if all(value is not None for value in values):
            complete_tracks.add(tuple(int(value) for value in values))
    return complete_tracks


def track_pair_set(
    track_matrix: Any, *, session_pairs: Iterable[tuple[int, int]] | None = None
) -> set[TrackLink]:
    """Return pairwise track links as ``(session_a, session_b, obs_a, obs_b)``."""
    matrix = normalize_track_matrix(track_matrix)
    links: set[TrackLink] = set()
    for session_a, session_b in _session_pairs(matrix, session_pairs):
        for row in matrix:
            obs_a = row[session_a]
            obs_b = row[session_b]
            if obs_a is not None and obs_b is not None:
                links.add((int(session_a), int(session_b), int(obs_a), int(obs_b)))
    return links


def pairwise_track_set(
    track_matrix: Any, *, session_pairs: Iterable[tuple[int, int]] | None = None
) -> set[TrackLink]:
    """Backward-compatible alias for :func:`track_pair_set`."""
    return track_pair_set(track_matrix, session_pairs=session_pairs)


def reference_fragment_counts(
    predicted_track_matrix: Any, reference_track_matrix: Any
) -> np.ndarray:
    """Return the number of predicted fragments covering each reference track."""
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)
    predicted_lookup = _multi_observation_lookup(predicted)
    counts = np.zeros(reference.shape[0], dtype=int)
    for reference_idx, row in enumerate(reference):
        covering_tracks: set[int] = set()
        for session_idx, observation in enumerate(row):
            if observation is not None:
                covering_tracks.update(
                    predicted_lookup.get((int(session_idx), int(observation)), ())
                )
        counts[reference_idx] = len(covering_tracks)
    return counts


def score_complete_tracks(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Score exact complete-track recovery with precision, recall, and F1."""
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)
    return _score_identity_sets(
        complete_track_set(predicted, session_indices=session_indices),
        complete_track_set(reference, session_indices=session_indices),
        prefix="complete_track",
        predicted_total_name="complete_tracks",
        reference_total_name="reference_complete_tracks",
    )


def score_track_links(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    """Score pairwise links induced by predicted and reference track matrices."""
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)
    return _score_identity_sets(
        track_pair_set(predicted, session_pairs=session_pairs),
        track_pair_set(reference, session_pairs=session_pairs),
        prefix="track_link",
        predicted_total_name="track_links",
        reference_total_name="reference_track_links",
    )


def score_pairwise_tracks(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    """Score pairwise links using BayesCaTrack-compatible metric names."""
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)
    return _score_identity_sets(
        track_pair_set(predicted, session_pairs=session_pairs),
        track_pair_set(reference, session_pairs=session_pairs),
        prefix="pairwise",
        predicted_total_name="pairwise_links",
        reference_total_name="reference_pairwise_links",
    )


def score_false_continuations(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    """Score predicted forward links that contradict the reference identity map."""
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)
    reference_lookup = _single_observation_lookup(reference)[0]
    valid_links: set[TrackLink] = set()
    false_links: set[TrackLink] = set()
    unknown_source_links: set[TrackLink] = set()
    for session_a, session_b in _session_pairs(predicted, session_pairs):
        for row in predicted:
            obs_a = row[session_a]
            obs_b = row[session_b]
            if obs_a is None or obs_b is None:
                continue
            link = (int(session_a), int(session_b), int(obs_a), int(obs_b))
            reference_track_idx = reference_lookup.get((int(session_a), int(obs_a)))
            if reference_track_idx is None:
                unknown_source_links.add(link)
                continue
            expected_obs_b = reference[reference_track_idx, int(session_b)]
            if expected_obs_b is None or int(obs_b) != int(expected_obs_b):
                false_links.add(link)
            else:
                valid_links.add(link)
    labeled = len(valid_links) + len(false_links)
    return {
        "false_continuations": len(false_links),
        "valid_continuations": len(valid_links),
        "labeled_predicted_continuations": labeled,
        "unknown_source_continuations": len(unknown_source_links),
        "false_continuation_rate": _zero_ratio(len(false_links), labeled),
    }


def score_track_fragmentation(
    predicted_track_matrix: Any, reference_track_matrix: Any
) -> dict[str, float | int]:
    """Score fragmentation of reference identities across predicted tracks."""
    reference = normalize_track_matrix(reference_track_matrix)
    counts = reference_fragment_counts(predicted_track_matrix, reference)
    valid_counts = counts[track_lengths(reference) > 0]
    reference_tracks = int(valid_counts.size)
    covered_mask = valid_counts > 0
    covered_tracks = int(np.count_nonzero(covered_mask))
    fragmented_tracks = int(np.count_nonzero(valid_counts > 1))
    events = int(np.sum(np.maximum(valid_counts - 1, 0), dtype=int))
    return {
        "fragmentation_reference_tracks": reference_tracks,
        "fragmentation_covered_reference_tracks": covered_tracks,
        "fragmentation_fragmented_reference_tracks": fragmented_tracks,
        "fragmentation_fragments": int(np.sum(valid_counts, dtype=int)),
        "fragmentation_events": events,
        "fragmentation_rate": _zero_ratio(fragmented_tracks, reference_tracks),
        "fragmentation_covered_rate": _zero_ratio(fragmented_tracks, covered_tracks),
        "fragmentation_mean_fragments_per_reference_track": _mean_or_zero(valid_counts),
        "fragmentation_mean_fragments_per_covered_reference_track": _mean_or_zero(
            valid_counts[covered_mask]
        ),
        "fragmentation_max_fragments_per_reference_track": (
            int(np.max(valid_counts)) if reference_tracks else 0
        ),
    }


def score_fragmentation(
    predicted_track_matrix: Any, reference_track_matrix: Any
) -> dict[str, float | int]:
    """Backward-compatible alias for :func:`score_track_fragmentation`."""
    return score_track_fragmentation(predicted_track_matrix, reference_track_matrix)


def summarize_tracks(track_matrix: Any) -> dict[str, float | int]:
    """Summarize the number and length of tracks."""
    matrix = normalize_track_matrix(track_matrix)
    lengths = track_lengths(matrix)
    return {
        "tracks": int(matrix.shape[0]),
        "mean_track_length": float(np.mean(lengths)) if lengths.size else 0.0,
        "max_track_length": int(np.max(lengths)) if lengths.size else 0,
    }


def track_error_ledger(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Return detailed track-, link-, and duplicate-observation error ledgers."""
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)
    pairs = _session_pairs(predicted, session_pairs)
    predicted_observations = _observations_by_track(predicted)
    reference_observations = _observations_by_track(reference)
    predicted_lookup, predicted_duplicates = _single_observation_lookup(predicted)
    reference_lookup, reference_duplicates = _single_observation_lookup(reference)
    predicted_links = track_pair_set(predicted, session_pairs=pairs)
    reference_links = track_pair_set(reference, session_pairs=pairs)
    false_links = sorted(predicted_links.difference(reference_links))
    missed_links = sorted(reference_links.difference(predicted_links))
    predicted_rows = _track_rows(
        predicted_observations, reference_lookup, pairs, predicted=True
    )
    reference_rows = _track_rows(
        reference_observations, predicted_lookup, pairs, predicted=False
    )
    link_rows = [
        _false_link_row(link, predicted_lookup, reference_lookup)
        for link in false_links
    ]
    link_rows.extend(
        _missed_link_row(link, predicted_lookup, reference_lookup)
        for link in missed_links
    )
    duplicate_rows = _duplicate_rows(
        "predicted", predicted_duplicates
    ) + _duplicate_rows("reference", reference_duplicates)
    return {
        "summary": _summary_rows(
            predicted_rows,
            reference_rows,
            false_links,
            missed_links,
            predicted_links,
            reference_links,
            predicted_duplicates,
            reference_duplicates,
        ),
        "predicted_tracks": predicted_rows,
        "reference_tracks": reference_rows,
        "link_errors": link_rows,
        "duplicate_observations": duplicate_rows,
    }


def summarize_track_errors(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, int | float]:
    """Return aggregate track-level error metrics."""
    return track_error_ledger(
        predicted_track_matrix, reference_track_matrix, session_pairs=session_pairs
    )["summary"]


def score_track_matrices(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
    complete_session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Return aggregate link, complete-track, fragmentation, and ledger metrics."""
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    _validate_compatible_shapes(predicted, reference)
    scores: dict[str, float | int] = {}
    scores.update(score_track_links(predicted, reference, session_pairs=session_pairs))
    scores.update(
        score_pairwise_tracks(predicted, reference, session_pairs=session_pairs)
    )
    scores.update(
        score_complete_tracks(
            predicted, reference, session_indices=complete_session_indices
        )
    )
    scores.update(
        score_false_continuations(predicted, reference, session_pairs=session_pairs)
    )
    scores.update(score_track_fragmentation(predicted, reference))
    scores.update(summarize_tracks(predicted))
    error_scores = dict(
        summarize_track_errors(predicted, reference, session_pairs=session_pairs)
    )
    false_continuation_link_rate = error_scores.pop("false_continuation_rate", None)
    scores.update(error_scores)
    if false_continuation_link_rate is not None:
        scores["false_continuation_link_rate"] = false_continuation_link_rate
    return scores


def _parse_optional_int(value: Any) -> int | None:
    candidate = _optional_int_candidate(value)
    if candidate is _MISSING:
        return None
    if (
        isinstance(candidate, (float, np.floating))
        and not float(candidate).is_integer()
    ):
        return None
    try:
        parsed = int(candidate)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_int_candidate(value: Any) -> Any:
    if value is None:
        return _MISSING
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        stripped = value.strip()
        return _MISSING if stripped.casefold() in _MISSING_STRINGS else stripped
    if isinstance(value, (float, np.floating)) and bool(np.isnan(value)):
        return _MISSING
    return value


def _selected_sessions(
    matrix: np.ndarray, session_indices: Sequence[int] | None
) -> list[int]:
    selected = (
        list(range(matrix.shape[1]))
        if session_indices is None
        else [int(index) for index in session_indices]
    )
    for session_idx in selected:
        _validate_session_index(matrix, session_idx)
    return selected


def _session_pairs(
    matrix: np.ndarray, session_pairs: Iterable[tuple[int, int]] | None
) -> tuple[tuple[int, int], ...]:
    pairs = (
        tuple((idx, idx + 1) for idx in range(max(0, matrix.shape[1] - 1)))
        if session_pairs is None
        else tuple((int(a), int(b)) for a, b in session_pairs)
    )
    for session_a, session_b in pairs:
        _validate_session_index(matrix, session_a)
        _validate_session_index(matrix, session_b)
        if session_a >= session_b:
            raise ValueError("session_pairs must point forward in time")
    return pairs


def _validate_session_index(matrix: np.ndarray, session_idx: int) -> None:
    if session_idx < 0 or session_idx >= matrix.shape[1]:
        raise IndexError(
            f"session index {session_idx} out of bounds for {matrix.shape[1]} sessions"
        )


def _validate_compatible_shapes(predicted: np.ndarray, reference: np.ndarray) -> None:
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError(
            "Predicted and reference matrices must have the same number of sessions"
        )


def _multi_observation_lookup(matrix: np.ndarray) -> dict[Observation, set[int]]:
    lookup: dict[Observation, set[int]] = {}
    for track_idx, row in enumerate(matrix):
        for session_idx, observation in enumerate(row):
            if observation is not None:
                lookup.setdefault((int(session_idx), int(observation)), set()).add(
                    int(track_idx)
                )
    return lookup


def _single_observation_lookup(
    matrix: np.ndarray,
) -> tuple[dict[Observation, int], list[tuple[Observation, int, int]]]:
    lookup: dict[Observation, int] = {}
    duplicates: list[tuple[Observation, int, int]] = []
    for track_idx, row in enumerate(matrix):
        for session_idx, observation in enumerate(row):
            if observation is None:
                continue
            key = (int(session_idx), int(observation))
            first_track_idx = lookup.get(key)
            if first_track_idx is None:
                lookup[key] = int(track_idx)
            else:
                duplicates.append((key, int(first_track_idx), int(track_idx)))
    return lookup, duplicates


def _observations_by_track(matrix: np.ndarray) -> list[list[Observation]]:
    return [
        [
            (int(session_idx), int(value))
            for session_idx, value in enumerate(row)
            if value is not None
        ]
        for row in matrix
    ]


def _track_rows(
    observations_by_track: list[list[Observation]],
    lookup: dict[Observation, int],
    pairs: tuple[tuple[int, int], ...],
    *,
    predicted: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track_id, observations in enumerate(observations_by_track):
        matched_ids = [
            lookup[observation] for observation in observations if observation in lookup
        ]
        counts = Counter(matched_ids)
        missing = [
            observation for observation in observations if observation not in lookup
        ]
        if predicted:
            switches = _identity_switches(observations, lookup)
            rows.append(
                {
                    "track_id": int(track_id),
                    "category": _predicted_track_category(counts, missing, switches),
                    "length": len(observations),
                    "reference_track_ids": sorted(int(track_id) for track_id in counts),
                    "dominant_reference_track_id": _dominant_counter_key(counts),
                    "matched_reference_observations": int(sum(counts.values())),
                    "unreferenced_observations": len(missing),
                    "identity_switches": switches,
                    "false_continuation_links": _track_disagreement_links(
                        observations, lookup, pairs
                    ),
                }
            )
        else:
            rows.append(
                {
                    "track_id": int(track_id),
                    "category": _reference_track_category(counts, missing),
                    "length": len(observations),
                    "predicted_track_ids": sorted(int(track_id) for track_id in counts),
                    "dominant_predicted_track_id": _dominant_counter_key(counts),
                    "matched_predicted_observations": int(sum(counts.values())),
                    "missed_observations": len(missing),
                    "fragment_count": max(0, len(counts) - 1),
                    "missed_reference_links": _track_disagreement_links(
                        observations, lookup, pairs
                    ),
                }
            )
    return rows


def _summary_rows(
    predicted_rows,
    reference_rows,
    false_links,
    missed_links,
    predicted_links,
    reference_links,
    predicted_duplicates,
    reference_duplicates,
):
    reference_tracks = len(reference_rows)
    predicted_tracks = len(predicted_rows)
    fragmented = sum(1 for row in reference_rows if row["fragment_count"] > 0)
    missed_tracks = sum(1 for row in reference_rows if row["category"] == "missed")
    partial_tracks = sum(
        1
        for row in reference_rows
        if row["category"] in {"partial", "fragmented_partial"}
    )
    mixed_tracks = sum(
        1 for row in predicted_rows if row["category"] == "mixed_identity"
    )
    spurious_tracks = sum(1 for row in predicted_rows if row["category"] == "spurious")
    predicted_error_tracks = sum(
        1 for row in predicted_rows if row["category"] != "single_identity"
    )
    reference_error_tracks = sum(
        1 for row in reference_rows if row["category"] != "recovered"
    )
    return {
        "identity_switches": int(
            sum(row["identity_switches"] for row in predicted_rows)
        ),
        "mixed_identity_tracks": int(mixed_tracks),
        "spurious_tracks": int(spurious_tracks),
        "spurious_predicted_observations": int(
            sum(row["unreferenced_observations"] for row in predicted_rows)
        ),
        "false_continuation_links": len(false_links),
        "false_continuation_rate": _zero_ratio(len(false_links), len(predicted_links)),
        "missed_reference_links": len(missed_links),
        "missed_reference_link_rate": _zero_ratio(
            len(missed_links), len(reference_links)
        ),
        "fragmented_reference_tracks": int(fragmented),
        "track_fragmentations": int(
            sum(row["fragment_count"] for row in reference_rows)
        ),
        "track_fragmentation_rate": _zero_ratio(fragmented, reference_tracks),
        "missed_reference_tracks": int(missed_tracks),
        "missed_reference_track_rate": _zero_ratio(missed_tracks, reference_tracks),
        "partial_reference_tracks": int(partial_tracks),
        "missed_reference_observations": int(
            sum(row["missed_observations"] for row in reference_rows)
        ),
        "predicted_duplicate_observations": len(predicted_duplicates),
        "reference_duplicate_observations": len(reference_duplicates),
        "predicted_tracks_with_errors": int(predicted_error_tracks),
        "reference_tracks_with_errors": int(reference_error_tracks),
        "predicted_track_error_rate": _zero_ratio(
            predicted_error_tracks, predicted_tracks
        ),
        "reference_track_error_rate": _zero_ratio(
            reference_error_tracks, reference_tracks
        ),
    }


def _false_link_row(
    link: TrackLink,
    predicted_lookup: dict[Observation, int],
    reference_lookup: dict[Observation, int],
) -> dict[str, Any]:
    session_a, session_b, obs_a, obs_b = link
    obs_key_a = (session_a, obs_a)
    obs_key_b = (session_b, obs_b)
    reference_a = reference_lookup.get(obs_key_a)
    reference_b = reference_lookup.get(obs_key_b)
    reason = (
        "unreferenced_observation"
        if reference_a is None or reference_b is None
        else (
            "different_reference_tracks"
            if reference_a != reference_b
            else "missing_reference_link"
        )
    )
    return {
        "error_type": "false_continuation",
        "reason": reason,
        "session_a": int(session_a),
        "session_b": int(session_b),
        "observation_a": int(obs_a),
        "observation_b": int(obs_b),
        "predicted_track_id": _same_or_none(
            predicted_lookup.get(obs_key_a), predicted_lookup.get(obs_key_b)
        ),
        "reference_track_a": _optional_int(reference_a),
        "reference_track_b": _optional_int(reference_b),
    }


def _missed_link_row(
    link: TrackLink,
    predicted_lookup: dict[Observation, int],
    reference_lookup: dict[Observation, int],
) -> dict[str, Any]:
    session_a, session_b, obs_a, obs_b = link
    obs_key_a = (session_a, obs_a)
    obs_key_b = (session_b, obs_b)
    predicted_a = predicted_lookup.get(obs_key_a)
    predicted_b = predicted_lookup.get(obs_key_b)
    reason = (
        "missing_prediction_observation"
        if predicted_a is None or predicted_b is None
        else (
            "split_across_predicted_tracks"
            if predicted_a != predicted_b
            else "missing_predicted_link"
        )
    )
    return {
        "error_type": "missed_reference_link",
        "reason": reason,
        "session_a": int(session_a),
        "session_b": int(session_b),
        "observation_a": int(obs_a),
        "observation_b": int(obs_b),
        "reference_track_id": _same_or_none(
            reference_lookup.get(obs_key_a), reference_lookup.get(obs_key_b)
        ),
        "predicted_track_a": _optional_int(predicted_a),
        "predicted_track_b": _optional_int(predicted_b),
    }


def _duplicate_rows(
    kind: str, duplicates: list[tuple[Observation, int, int]]
) -> list[dict[str, int | str]]:
    return [
        {
            "matrix": kind,
            "session": int(obs[0]),
            "observation": int(obs[1]),
            "first_track_id": int(first),
            "duplicate_track_id": int(duplicate),
        }
        for obs, first, duplicate in duplicates
    ]


def _track_disagreement_links(
    observations: list[Observation],
    lookup: dict[Observation, int],
    pairs: tuple[tuple[int, int], ...],
) -> int:
    by_session = dict(observations)
    errors = 0
    for session_a, session_b in pairs:
        if session_a not in by_session or session_b not in by_session:
            continue
        track_a = lookup.get((session_a, by_session[session_a]))
        track_b = lookup.get((session_b, by_session[session_b]))
        if track_a is None or track_b is None or track_a != track_b:
            errors += 1
    return errors


def _identity_switches(
    observations: list[Observation], lookup: dict[Observation, int]
) -> int:
    last_id: int | None = None
    switches = 0
    for observation in sorted(observations):
        track_id = lookup.get(observation)
        if track_id is None:
            continue
        if last_id is not None and track_id != last_id:
            switches += 1
        last_id = track_id
    return switches


def _predicted_track_category(
    reference_counts: Counter[int],
    unknown_observations: list[Observation],
    identity_switches: int,
) -> str:
    if not reference_counts:
        return "spurious"
    if identity_switches > 0 or len(reference_counts) > 1:
        return "mixed_identity"
    return (
        "single_identity_with_unreferenced_observations"
        if unknown_observations
        else "single_identity"
    )


def _reference_track_category(
    predicted_counts: Counter[int], missed_observations: list[Observation]
) -> str:
    if not predicted_counts:
        return "missed"
    fragmented = len(predicted_counts) > 1
    partial = bool(missed_observations)
    if fragmented and partial:
        return "fragmented_partial"
    if fragmented:
        return "fragmented"
    return "partial" if partial else "recovered"


def _score_identity_sets(
    predicted: set[Any],
    reference: set[Any],
    *,
    prefix: str,
    predicted_total_name: str,
    reference_total_name: str,
) -> dict[str, float | int]:
    true_positives = len(predicted & reference)
    false_positives = len(predicted - reference)
    false_negatives = len(reference - predicted)
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        f"{prefix}_true_positives": true_positives,
        f"{prefix}_false_positives": false_positives,
        f"{prefix}_false_negatives": false_negatives,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        predicted_total_name: len(predicted),
        reference_total_name: len(reference),
    }


def _dominant_counter_key(counter: Counter[int]) -> int | None:
    return (
        None if not counter else int(max(counter, key=lambda key: (counter[key], -key)))
    )


def _same_or_none(left: int | None, right: int | None) -> int | None:
    return int(left) if left is not None and left == right else None


def _optional_int(value: int | None) -> int | None:
    return None if value is None else int(value)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 1.0 if denominator == 0 else float(numerator) / float(denominator)


def _zero_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _mean_or_zero(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0
