"""Track-level outcome metrics built on track matrices."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np

from .track_evaluation import normalize_track_matrix, score_track_fragmentation

Observation: TypeAlias = tuple[int, int]

__all__ = [
    "false_track_rate",
    "missed_track_rate",
    "score_false_tracks",
    "score_missed_tracks",
    "score_track_latency",
    "score_track_outcomes",
    "score_track_purity",
    "track_latencies",
    "track_purity",
]


def score_track_purity(
    predicted_track_matrix: Any, reference_track_matrix: Any
) -> dict[str, float | int]:
    """Return predicted-track identity purity metrics.

    A predicted track's purity is the fraction of its observations that belong
    to its dominant reference identity. Observations absent from the reference
    matrix count as impure in ``mean_track_purity`` and
    ``observation_weighted_track_purity``.
    """
    predicted, reference = _normalized_pair(
        predicted_track_matrix, reference_track_matrix
    )
    reference_lookup = _single_observation_lookup(reference)

    purities: list[float] = []
    labeled_purities: list[float] = []
    dominant_observations = 0
    labeled_observations = 0
    total_observations = 0
    unreferenced_observations = 0
    pure_tracks = 0
    impure_tracks = 0
    empty_tracks = 0

    for observations in _observations_by_track(predicted):
        if not observations:
            empty_tracks += 1
            continue
        total_observations += len(observations)
        matched_reference_ids = [
            reference_lookup[observation]
            for observation in observations
            if observation in reference_lookup
        ]
        counts = Counter(matched_reference_ids)
        labeled_count = int(sum(counts.values()))
        dominant_count = max(counts.values(), default=0)
        labeled_observations += labeled_count
        dominant_observations += dominant_count
        unreferenced_observations += len(observations) - labeled_count
        purities.append(_zero_ratio(dominant_count, len(observations)))
        if labeled_count:
            labeled_purities.append(_zero_ratio(dominant_count, labeled_count))
        if dominant_count == len(observations):
            pure_tracks += 1
        else:
            impure_tracks += 1

    return {
        "track_purity_tracks": int(len(purities)),
        "track_purity_empty_tracks": int(empty_tracks),
        "pure_tracks": int(pure_tracks),
        "impure_tracks": int(impure_tracks),
        "mean_track_purity": _mean_or_zero(purities),
        "observation_weighted_track_purity": _zero_ratio(
            dominant_observations, total_observations
        ),
        "mean_labeled_track_purity": _mean_or_zero(labeled_purities),
        "observation_weighted_labeled_track_purity": _zero_ratio(
            dominant_observations, labeled_observations
        ),
        "unreferenced_predicted_observations": int(unreferenced_observations),
        "unreferenced_observation_rate": _zero_ratio(
            unreferenced_observations, total_observations
        ),
    }


def track_purity(predicted_track_matrix: Any, reference_track_matrix: Any) -> float:
    """Return observation-weighted predicted-track purity."""
    return float(
        score_track_purity(predicted_track_matrix, reference_track_matrix)[
            "observation_weighted_track_purity"
        ]
    )


def score_false_tracks(
    predicted_track_matrix: Any, reference_track_matrix: Any, *, min_length: int = 1
) -> dict[str, float | int]:
    """Return metrics for predicted tracks that contain no reference observation."""
    if int(min_length) <= 0:
        raise ValueError("min_length must be positive")
    predicted, reference = _normalized_pair(
        predicted_track_matrix, reference_track_matrix
    )
    reference_lookup = _single_observation_lookup(reference)

    evaluated_tracks = 0
    false_tracks = 0
    false_track_observations = 0
    unreferenced_observations = 0
    total_observations = 0
    for observations in _observations_by_track(predicted):
        total_observations += len(observations)
        if len(observations) < int(min_length):
            continue
        evaluated_tracks += 1
        matched = sum(
            1 for observation in observations if observation in reference_lookup
        )
        unreferenced_observations += len(observations) - matched
        if matched == 0:
            false_tracks += 1
            false_track_observations += len(observations)

    return {
        "false_tracks": int(false_tracks),
        "false_track_evaluated_tracks": int(evaluated_tracks),
        "false_track_rate": _zero_ratio(false_tracks, evaluated_tracks),
        "false_track_observations": int(false_track_observations),
        "unreferenced_predicted_observations": int(unreferenced_observations),
        "unreferenced_predicted_observation_rate": _zero_ratio(
            unreferenced_observations, total_observations
        ),
    }


def false_track_rate(
    predicted_track_matrix: Any, reference_track_matrix: Any, *, min_length: int = 1
) -> float:
    """Return the fraction of evaluated predicted tracks that are false."""
    return float(
        score_false_tracks(
            predicted_track_matrix, reference_track_matrix, min_length=min_length
        )["false_track_rate"]
    )


def score_missed_tracks(
    predicted_track_matrix: Any, reference_track_matrix: Any, *, min_length: int = 1
) -> dict[str, float | int]:
    """Return metrics for reference tracks with no predicted observation support."""
    if int(min_length) <= 0:
        raise ValueError("min_length must be positive")
    predicted, reference = _normalized_pair(
        predicted_track_matrix, reference_track_matrix
    )
    predicted_lookup = _single_observation_lookup(predicted)

    evaluated_tracks = 0
    missed_tracks = 0
    covered_tracks = 0
    missed_observations = 0
    total_observations = 0
    for observations in _observations_by_track(reference):
        total_observations += len(observations)
        if len(observations) < int(min_length):
            continue
        evaluated_tracks += 1
        recovered = sum(
            1 for observation in observations if observation in predicted_lookup
        )
        missed_observations += len(observations) - recovered
        if recovered == 0:
            missed_tracks += 1
        else:
            covered_tracks += 1

    return {
        "missed_tracks": int(missed_tracks),
        "covered_reference_tracks": int(covered_tracks),
        "missed_track_evaluated_reference_tracks": int(evaluated_tracks),
        "missed_track_rate": _zero_ratio(missed_tracks, evaluated_tracks),
        "missed_reference_observations": int(missed_observations),
        "missed_reference_observation_rate": _zero_ratio(
            missed_observations, total_observations
        ),
    }


def missed_track_rate(
    predicted_track_matrix: Any, reference_track_matrix: Any, *, min_length: int = 1
) -> float:
    """Return the fraction of evaluated reference tracks that are missed."""
    return float(
        score_missed_tracks(
            predicted_track_matrix, reference_track_matrix, min_length=min_length
        )["missed_track_rate"]
    )


def track_latencies(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_times: Sequence[float] | None = None,
    missed_value=np.nan,
) -> np.ndarray:
    """Return first-detection latency for each non-empty reference track."""
    predicted, reference = _normalized_pair(
        predicted_track_matrix, reference_track_matrix
    )
    predicted_lookup = _single_observation_lookup(predicted)
    times = _session_times(reference.shape[1], session_times)
    values: list[float] = []
    for observations in _observations_by_track(reference):
        if not observations:
            continue
        first_reference_session = min(session for session, _ in observations)
        detected_sessions = [
            session
            for session, observation in observations
            if (session, observation) in predicted_lookup
        ]
        if not detected_sessions:
            values.append(float(missed_value))
            continue
        values.append(
            float(times[min(detected_sessions)] - times[first_reference_session])
        )
    return np.asarray(values, dtype=float)


def score_track_latency(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_times: Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Return aggregate first-detection latency metrics."""
    values = track_latencies(
        predicted_track_matrix, reference_track_matrix, session_times=session_times
    )
    detected_values = values[np.isfinite(values)]
    return {
        "latency_reference_tracks": int(values.size),
        "latency_detected_tracks": int(detected_values.size),
        "latency_missed_tracks": int(values.size - detected_values.size),
        "track_detection_rate": _zero_ratio(detected_values.size, values.size),
        "mean_track_latency": _mean_or_zero(detected_values),
        "median_track_latency": (
            float(np.median(detected_values)) if detected_values.size else 0.0
        ),
        "max_track_latency": (
            float(np.max(detected_values)) if detected_values.size else 0.0
        ),
    }


def score_track_outcomes(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_times: Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Return fragmentation, purity, false-track, missed-track, and latency metrics."""
    scores: dict[str, float | int] = {}
    scores.update(
        score_track_fragmentation(predicted_track_matrix, reference_track_matrix)
    )
    scores.update(score_track_purity(predicted_track_matrix, reference_track_matrix))
    scores.update(score_false_tracks(predicted_track_matrix, reference_track_matrix))
    scores.update(score_missed_tracks(predicted_track_matrix, reference_track_matrix))
    scores.update(
        score_track_latency(
            predicted_track_matrix, reference_track_matrix, session_times=session_times
        )
    )
    return scores


def _normalized_pair(
    predicted_track_matrix: Any, reference_track_matrix: Any
) -> tuple[np.ndarray, np.ndarray]:
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError(
            "Predicted and reference matrices must have the same number of sessions"
        )
    return predicted, reference


def _observations_by_track(matrix: np.ndarray) -> list[list[Observation]]:
    return [
        [
            (int(session_idx), int(value))
            for session_idx, value in enumerate(row)
            if value is not None
        ]
        for row in matrix
    ]


def _single_observation_lookup(matrix: np.ndarray) -> dict[Observation, int]:
    lookup: dict[Observation, int] = {}
    for track_idx, row in enumerate(matrix):
        for session_idx, observation in enumerate(row):
            if observation is not None:
                lookup.setdefault((int(session_idx), int(observation)), int(track_idx))
    return lookup


def _session_times(
    n_sessions: int, session_times: Sequence[float] | None
) -> np.ndarray:
    if session_times is None:
        return np.arange(n_sessions, dtype=float)
    times = np.asarray(session_times, dtype=float)
    if times.shape != (n_sessions,):
        raise ValueError(
            "session_times must have length equal to the number of sessions"
        )
    return times


def _zero_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _mean_or_zero(values: Sequence[float] | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0
