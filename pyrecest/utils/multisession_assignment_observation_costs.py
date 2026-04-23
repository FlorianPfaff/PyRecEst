"""Extensions for multi-session assignment with observation-specific start/end costs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pyrecest.backend import all, asarray, cast, full, isfinite  # pylint: disable=no-name-in-module
from pyrecest.backend import max as backend_max
from pyrecest.backend import sum as backend_sum

from .multisession_assignment import (
    MultiSessionAssignmentResult,
    solve_multisession_assignment,
)

BackendArray = Any
PairwiseCostsInput = Mapping[tuple[int, int], Any] | Sequence[Any]
SessionSizesInput = Mapping[int, int] | Sequence[int]
Observation = tuple[int, int]
ObservationCostValue = float | Sequence[float] | BackendArray
ObservationCostsInput = Mapping[int, ObservationCostValue] | Sequence[ObservationCostValue]


def solve_multisession_assignment_with_observation_costs(
    pairwise_costs: PairwiseCostsInput,
    session_sizes: SessionSizesInput | None = None,
    *,
    start_cost: float = 0.0,
    end_cost: float = 0.0,
    start_costs: ObservationCostsInput | None = None,
    end_costs: ObservationCostsInput | None = None,
    gap_penalty: float = 0.0,
    cost_threshold: float | None = None,
) -> MultiSessionAssignmentResult:
    """Solve global multi-session assignment with heterogeneous birth/death costs.

    This is a thin wrapper around :func:`solve_multisession_assignment`. It encodes
    observation-specific start/end costs into transformed pairwise costs and then
    maps the reported objective value and matched-edge costs back to the original
    problem.
    """

    normalized_pairwise = _normalize_pairwise_costs(pairwise_costs)
    normalized_sizes = _normalize_session_sizes(session_sizes)
    session_sizes_map = _infer_and_validate_session_sizes(
        normalized_pairwise,
        normalized_sizes,
    )

    if not session_sizes_map:
        return MultiSessionAssignmentResult(tracks=[], matched_edges=[], total_cost=0.0)

    normalized_start_costs = _normalize_observation_costs(
        start_costs,
        session_sizes_map,
        default_value=float(start_cost),
        name="start_costs",
    )
    normalized_end_costs = _normalize_observation_costs(
        end_costs,
        session_sizes_map,
        default_value=float(end_cost),
        name="end_costs",
    )

    max_start_cost = max(
        float(backend_max(costs)) if costs.size else float(start_cost)
        for costs in normalized_start_costs.values()
    )
    max_end_cost = max(
        float(backend_max(costs)) if costs.size else float(end_cost)
        for costs in normalized_end_costs.values()
    )

    session_order = sorted(session_sizes_map)
    session_positions = {
        session_idx: position for position, session_idx in enumerate(session_order)
    }

    transformed_pairwise = _transform_pairwise_costs(
        normalized_pairwise,
        session_positions,
        normalized_start_costs,
        normalized_end_costs,
        uniform_start_cost=max_start_cost,
        uniform_end_cost=max_end_cost,
        gap_penalty=float(gap_penalty),
        cost_threshold=cost_threshold,
    )

    base_result = solve_multisession_assignment(
        transformed_pairwise,
        session_sizes=session_sizes_map,
        start_cost=max_start_cost,
        end_cost=max_end_cost,
        gap_penalty=gap_penalty,
        cost_threshold=None,
    )

    actual_baseline = sum(
        float(backend_sum(costs)) for costs in normalized_start_costs.values()
    ) + sum(float(backend_sum(costs)) for costs in normalized_end_costs.values())
    uniform_baseline = sum(session_sizes_map.values()) * (max_start_cost + max_end_cost)
    total_cost = float(base_result.total_cost - uniform_baseline + actual_baseline)

    adjusted_edges = [
        (
            source,
            target,
            float(
                transformed_cost
                - max_end_cost
                - max_start_cost
                + normalized_end_costs[source[0]][source[1]]
                + normalized_start_costs[target[0]][target[1]]
            ),
        )
        for source, target, transformed_cost in base_result.matched_edges
    ]

    return MultiSessionAssignmentResult(
        tracks=base_result.tracks,
        matched_edges=adjusted_edges,
        total_cost=total_cost,
    )


def _normalize_pairwise_costs(
    pairwise_costs: PairwiseCostsInput,
) -> dict[tuple[int, int], BackendArray]:
    if isinstance(pairwise_costs, Mapping):
        normalized: dict[tuple[int, int], BackendArray] = {}
        for key, value in pairwise_costs.items():
            if len(key) != 2:
                raise ValueError("Each pairwise-cost key must contain two session indices.")
            source_session, target_session = int(key[0]), int(key[1])
            if source_session >= target_session:
                raise ValueError("Pairwise-cost keys must satisfy source_session < target_session.")
            matrix = asarray(value, dtype=float)
            if matrix.ndim != 2:
                raise ValueError("Each pairwise cost matrix must be two-dimensional.")
            normalized[(source_session, target_session)] = matrix
        return normalized

    normalized = {}
    for session_idx, value in enumerate(pairwise_costs):
        matrix = asarray(value, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Each pairwise cost matrix must be two-dimensional.")
        normalized[(session_idx, session_idx + 1)] = matrix
    return normalized


def _normalize_session_sizes(
    session_sizes: SessionSizesInput | None,
) -> dict[int, int]:
    if session_sizes is None:
        return {}
    if isinstance(session_sizes, Mapping):
        normalized = {int(session_idx): int(size) for session_idx, size in session_sizes.items()}
    else:
        normalized = {session_idx: int(size) for session_idx, size in enumerate(session_sizes)}
    for session_idx, size in normalized.items():
        if size < 0:
            raise ValueError(f"Session {session_idx} has a negative detection count.")
    return normalized


def _infer_and_validate_session_sizes(
    pairwise_costs: Mapping[tuple[int, int], BackendArray],
    session_sizes: Mapping[int, int],
) -> dict[int, int]:
    inferred_sizes = dict(session_sizes)
    for (source_session, target_session), cost_matrix in pairwise_costs.items():
        source_size, target_size = cost_matrix.shape
        _check_or_set_session_size(inferred_sizes, source_session, source_size)
        _check_or_set_session_size(inferred_sizes, target_session, target_size)
    if not inferred_sizes and not pairwise_costs:
        raise ValueError("No observations were provided. Supply pairwise_costs or session_sizes.")
    return dict(sorted(inferred_sizes.items()))


def _check_or_set_session_size(
    inferred_sizes: dict[int, int],
    session_idx: int,
    candidate_size: int,
) -> None:
    if session_idx in inferred_sizes and inferred_sizes[session_idx] != candidate_size:
        raise ValueError(
            f"Inconsistent detection count for session {session_idx}: "
            f"expected {inferred_sizes[session_idx]}, got {candidate_size}."
        )
    inferred_sizes[session_idx] = int(candidate_size)


def _normalize_observation_costs(
    observation_costs: ObservationCostsInput | None,
    session_sizes: Mapping[int, int],
    *,
    default_value: float,
    name: str,
) -> dict[int, BackendArray]:
    if observation_costs is None:
        raw_entries: dict[int, ObservationCostValue] = {}
    elif isinstance(observation_costs, Mapping):
        raw_entries = {int(session_idx): value for session_idx, value in observation_costs.items()}
    else:
        raw_entries = {session_idx: value for session_idx, value in enumerate(observation_costs)}

    unknown_sessions = sorted(set(raw_entries) - set(session_sizes))
    if unknown_sessions:
        raise ValueError(
            f"{name} specifies unknown sessions {unknown_sessions}; "
            f"known sessions are {sorted(session_sizes)}."
        )

    normalized: dict[int, BackendArray] = {}
    for session_idx in sorted(session_sizes):
        session_size = int(session_sizes[session_idx])
        if session_idx not in raw_entries:
            values = full(session_size, float(default_value), dtype=float)
        else:
            values = _normalize_observation_cost_entry(
                raw_entries[session_idx],
                session_size,
                session_idx=session_idx,
                name=name,
            )
        if not all(isfinite(values)):
            raise ValueError(f"{name} for session {session_idx} must contain only finite values.")
        normalized[session_idx] = values
    return normalized


def _normalize_observation_cost_entry(
    value: ObservationCostValue,
    session_size: int,
    *,
    session_idx: int,
    name: str,
) -> BackendArray:
    values = asarray(value, dtype=float)
    if values.ndim == 0:
        return full(session_size, float(values), dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"{name} entry for session {session_idx} must be a scalar or a one-dimensional array."
        )
    if int(values.size) != int(session_size):
        raise ValueError(
            f"{name} entry for session {session_idx} has length {values.size}, expected {session_size}."
        )
    return cast(values, float)


def _transform_pairwise_costs(
    pairwise_costs: Mapping[tuple[int, int], BackendArray],
    session_positions: Mapping[int, int],
    start_costs: Mapping[int, BackendArray],
    end_costs: Mapping[int, BackendArray],
    *,
    uniform_start_cost: float,
    uniform_end_cost: float,
    gap_penalty: float,
    cost_threshold: float | None,
) -> dict[tuple[int, int], BackendArray]:
    transformed: dict[tuple[int, int], BackendArray] = {}
    for (source_session, target_session), matrix in pairwise_costs.items():
        source_position = session_positions[source_session]
        target_position = session_positions[target_session]
        gap = target_position - source_position - 1
        if gap < 0:
            raise ValueError("Session indices must define a forward-in-time edge ordering.")

        source_end = end_costs[source_session][:, None]
        target_start = start_costs[target_session][None, :]
        transformed_matrix = (
            asarray(matrix, dtype=float)
            - source_end
            - target_start
            + float(uniform_end_cost)
            + float(uniform_start_cost)
        )

        if cost_threshold is not None:
            adjusted_original = asarray(matrix, dtype=float) + float(gap_penalty) * gap
            transformed_matrix = transformed_matrix.copy()
            transformed_matrix[adjusted_original > float(cost_threshold)] = float("inf")

        transformed[(source_session, target_session)] = transformed_matrix
    return transformed


__all__ = ["solve_multisession_assignment_with_observation_costs"]
