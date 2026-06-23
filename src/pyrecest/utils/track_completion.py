"""Generic fragment completion candidates for multi-session track matrices.

The utilities in this module enumerate possible prefix/suffix continuations of
incomplete track rows.  They intentionally know nothing about a concrete sensor,
image representation, likelihood model, or tracker.  Domain projects provide a
``candidate_provider`` callback that maps an endpoint observation to possible
observations in another session; PyRecEst handles fragment discovery, recursive
path enumeration, duplicate-observation checks, and path bookkeeping.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar

import numpy as np

from .track_evaluation import normalize_track_matrix

CompletionDirection: TypeAlias = Literal["prefix", "suffix", "both"]
_DirectionalCompletion: TypeAlias = Literal["prefix", "suffix"]
PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class CompletionCandidate(Generic[PayloadT]):
    """One candidate observation returned by a domain-specific provider.

    Parameters
    ----------
    observation:
        Observation index in the candidate session.
    score:
        Optional additive score or cost contribution for the step.  The generic
        enumerator only sums these values unless ``score_path`` is supplied.
    payload:
        Optional domain-specific payload retained in the emitted step.
    """

    observation: int
    score: float = 0.0
    payload: PayloadT | None = None


@dataclass(frozen=True)
class CompletionStep(Generic[PayloadT]):
    """One chronological edge in a fragment-completion path."""

    from_session: int
    from_observation: int
    to_session: int
    to_observation: int
    score: float = 0.0
    payload: PayloadT | None = None


@dataclass(frozen=True)
class CompletionPath(Generic[PayloadT]):
    """A candidate prefix or suffix completion path for one predicted row."""

    track_index: int
    direction: _DirectionalCompletion
    steps: tuple[CompletionStep[PayloadT], ...]
    score: float

    @property
    def start_session(self) -> int:
        return self.steps[0].from_session

    @property
    def start_observation(self) -> int:
        return self.steps[0].from_observation

    @property
    def end_session(self) -> int:
        return self.steps[-1].to_session

    @property
    def end_observation(self) -> int:
        return self.steps[-1].to_observation

    @property
    def path_length(self) -> int:
        return len(self.steps)


CandidateProvider: TypeAlias = Callable[
    [int, int, int], Iterable[int | CompletionCandidate[PayloadT]]
]
CandidateSessionProvider: TypeAlias = Callable[
    [int, int, _DirectionalCompletion], Iterable[int]
]


def enumerate_fragment_completion_paths(
    track_matrix: Any,
    *,
    max_path_length: int = 2,
    direction: CompletionDirection = "both",
    candidate_provider: CandidateProvider[PayloadT],
    candidate_session_provider: CandidateSessionProvider | None = None,
    allow_duplicate_source: bool = False,
    allow_duplicate_target: bool = False,
    score_path: Callable[[tuple[CompletionStep[PayloadT], ...]], float] | None = None,
    max_paths_per_fragment: int | None = None,
) -> list[CompletionPath[PayloadT]]:
    """Enumerate candidate prefix/suffix completions for incomplete track rows.

    Parameters
    ----------
    track_matrix:
        Matrix-like object with rows as tracks and columns as sessions.  Missing
        values are parsed by :func:`pyrecest.utils.normalize_track_matrix`.
    max_path_length:
        Maximum number of candidate edges per emitted path.
    direction:
        Which fragment endpoints to complete.  ``"suffix"`` extends after the
        last observation in a row, ``"prefix"`` extends before the first
        observation, and ``"both"`` does both.
    candidate_provider:
        Callback ``candidate_provider(anchor_session, anchor_observation,
        candidate_session)`` returning observations in ``candidate_session``.
        For suffix completion the candidate session is after the anchor; for
        prefix completion it is before the anchor.  Returned values may be raw
        integer observation ids or :class:`CompletionCandidate` objects.
    candidate_session_provider:
        Optional callback returning candidate session indices for an endpoint.
        If omitted, only the adjacent previous/next session is considered.
    allow_duplicate_source / allow_duplicate_target:
        Whether paths may reuse already-occupied observations.  In ordinary
        track completion both should remain false.
    score_path:
        Optional aggregate scorer for a tuple of chronological steps.  If
        omitted, path scores are the sum of step scores.
    max_paths_per_fragment:
        Optional post-enumeration cap per endpoint, keeping the lowest-score
        paths first.  This is intended for diagnostics and small candidate sets;
        high-branching production trackers should pre-prune in the provider.
    """

    if max_path_length < 1:
        raise ValueError("max_path_length must be positive")
    if direction not in {"prefix", "suffix", "both"}:
        raise ValueError("direction must be 'prefix', 'suffix', or 'both'")
    if max_paths_per_fragment is not None and int(max_paths_per_fragment) < 1:
        raise ValueError("max_paths_per_fragment must be positive or None")

    matrix = normalize_track_matrix(track_matrix)
    occupied = occupied_observations_by_session(matrix)
    paths: list[CompletionPath[PayloadT]] = []
    directions: tuple[_DirectionalCompletion, ...]
    directions = ("prefix", "suffix") if direction == "both" else (direction,)

    for track_index, row in enumerate(matrix):
        present_sessions = [
            int(index) for index, value in enumerate(row) if value is not None
        ]
        if not present_sessions:
            continue
        for completion_direction in directions:
            if completion_direction == "suffix":
                endpoint_session = present_sessions[-1]
                if endpoint_session >= matrix.shape[1] - 1:
                    continue
            else:
                endpoint_session = present_sessions[0]
                if endpoint_session <= 0:
                    continue
            endpoint_observation = int(row[endpoint_session])
            fragment_paths: list[CompletionPath[PayloadT]] = []
            _extend_completion_path(
                matrix=matrix,
                track_index=int(track_index),
                direction=completion_direction,
                anchor_session=int(endpoint_session),
                anchor_observation=endpoint_observation,
                steps=(),
                seen={(int(endpoint_session), endpoint_observation)},
                occupied=occupied,
                max_path_length=int(max_path_length),
                candidate_provider=candidate_provider,
                candidate_session_provider=candidate_session_provider,
                allow_duplicate_source=bool(allow_duplicate_source),
                allow_duplicate_target=bool(allow_duplicate_target),
                score_path=score_path,
                out=fragment_paths,
            )
            if max_paths_per_fragment is not None:
                fragment_paths = sorted(
                    fragment_paths,
                    key=lambda path: (
                        path.score,
                        path.path_length,
                        path_observations(path),
                    ),
                )[: int(max_paths_per_fragment)]
            paths.extend(fragment_paths)
    return paths


def occupied_observations_by_session(track_matrix: Any) -> dict[tuple[int, int], int]:
    """Return occupancy counts keyed by ``(session, observation)``."""

    matrix = normalize_track_matrix(track_matrix)
    counts: dict[tuple[int, int], int] = {}
    for _, session_index in np.ndindex(matrix.shape):
        value = matrix[_, session_index]
        if value is not None:
            key = (int(session_index), int(value))
            counts[key] = counts.get(key, 0) + 1
    return counts


def path_sessions(path: CompletionPath[Any]) -> tuple[int, ...]:
    """Return chronological session indices in a completion path."""

    return (path.steps[0].from_session, *(step.to_session for step in path.steps))


def path_observations(path: CompletionPath[Any]) -> tuple[int, ...]:
    """Return chronological observation ids in a completion path."""

    return (
        path.steps[0].from_observation,
        *(step.to_observation for step in path.steps),
    )


def _extend_completion_path(
    *,
    matrix: np.ndarray,
    track_index: int,
    direction: _DirectionalCompletion,
    anchor_session: int,
    anchor_observation: int,
    steps: tuple[CompletionStep[PayloadT], ...],
    seen: set[tuple[int, int]],
    occupied: dict[tuple[int, int], int],
    max_path_length: int,
    candidate_provider: CandidateProvider[PayloadT],
    candidate_session_provider: CandidateSessionProvider | None,
    allow_duplicate_source: bool,
    allow_duplicate_target: bool,
    score_path: Callable[[tuple[CompletionStep[PayloadT], ...]], float] | None,
    out: list[CompletionPath[PayloadT]],
) -> None:
    if len(steps) >= max_path_length:
        return
    for candidate_session in _candidate_sessions(
        matrix,
        anchor_session,
        anchor_observation,
        direction,
        candidate_session_provider,
    ):
        if (
            not allow_duplicate_source
            and occupied.get((anchor_session, anchor_observation), 0) > 1
        ):
            continue
        for raw_candidate in candidate_provider(
            int(anchor_session), int(anchor_observation), int(candidate_session)
        ):
            candidate = _coerce_candidate(raw_candidate)
            candidate_key = (int(candidate_session), int(candidate.observation))
            if candidate_key in seen:
                continue
            if not allow_duplicate_target and occupied.get(candidate_key, 0) > 0:
                continue
            step = _completion_step(
                direction,
                anchor_session,
                anchor_observation,
                int(candidate_session),
                int(candidate.observation),
                float(candidate.score),
                candidate.payload,
            )
            raw_steps = (*steps, step)
            chronological_steps = _chronological_steps(direction, raw_steps)
            score = (
                float(score_path(chronological_steps))
                if score_path is not None
                else float(sum(item.score for item in chronological_steps))
            )
            out.append(
                CompletionPath(
                    track_index=int(track_index),
                    direction=direction,
                    steps=chronological_steps,
                    score=score,
                )
            )
            _extend_completion_path(
                matrix=matrix,
                track_index=track_index,
                direction=direction,
                anchor_session=int(candidate_session),
                anchor_observation=int(candidate.observation),
                steps=raw_steps,
                seen={*seen, candidate_key},
                occupied=occupied,
                max_path_length=max_path_length,
                candidate_provider=candidate_provider,
                candidate_session_provider=candidate_session_provider,
                allow_duplicate_source=allow_duplicate_source,
                allow_duplicate_target=allow_duplicate_target,
                score_path=score_path,
                out=out,
            )


def _candidate_sessions(
    matrix: np.ndarray,
    anchor_session: int,
    anchor_observation: int,
    direction: _DirectionalCompletion,
    provider: CandidateSessionProvider | None,
) -> tuple[int, ...]:
    if provider is None:
        adjacent = anchor_session + 1 if direction == "suffix" else anchor_session - 1
        raw = (adjacent,)
    else:
        raw = tuple(provider(anchor_session, anchor_observation, direction))
    sessions = []
    for raw_value in raw:
        value = _coerce_candidate_session(raw_value)
        if value is None:
            continue
        if value < 0 or value >= matrix.shape[1] or value == anchor_session:
            continue
        if direction == "suffix" and value <= anchor_session:
            continue
        if direction == "prefix" and value >= anchor_session:
            continue
        sessions.append(int(value))
    return tuple(dict.fromkeys(sessions))


def _coerce_candidate_session(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return int(value)
        return None
    return None


def _completion_step(
    direction: _DirectionalCompletion,
    anchor_session: int,
    anchor_observation: int,
    candidate_session: int,
    candidate_observation: int,
    score: float,
    payload: PayloadT | None,
) -> CompletionStep[PayloadT]:
    if direction == "suffix":
        return CompletionStep(
            from_session=anchor_session,
            from_observation=anchor_observation,
            to_session=candidate_session,
            to_observation=candidate_observation,
            score=score,
            payload=payload,
        )
    return CompletionStep(
        from_session=candidate_session,
        from_observation=candidate_observation,
        to_session=anchor_session,
        to_observation=anchor_observation,
        score=score,
        payload=payload,
    )


def _chronological_steps(
    direction: _DirectionalCompletion,
    steps: Sequence[CompletionStep[PayloadT]],
) -> tuple[CompletionStep[PayloadT], ...]:
    return tuple(steps if direction == "suffix" else reversed(steps))


def _coerce_candidate(
    value: int | CompletionCandidate[PayloadT],
) -> CompletionCandidate[PayloadT]:
    if isinstance(value, CompletionCandidate):
        return CompletionCandidate(
            observation=_normalize_candidate_observation(value.observation),
            score=value.score,
            payload=value.payload,
        )
    return CompletionCandidate(observation=_normalize_candidate_observation(value))


def _normalize_candidate_observation(value: Any) -> int:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError("candidate observations must be non-negative integers")

    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError("candidate observations must be non-negative integers")
    if isinstance(scalar, (int, np.integer)):
        observation = int(scalar)
    elif isinstance(scalar, (float, np.floating)):
        if not np.isfinite(scalar) or not float(scalar).is_integer():
            raise ValueError("candidate observations must be non-negative integers")
        observation = int(scalar)
    else:
        try:
            observation = int(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "candidate observations must be non-negative integers"
            ) from exc
    if observation < 0:
        raise ValueError("candidate observations must be non-negative integers")
    return observation
