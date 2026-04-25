# pylint: disable=duplicate-code
"""Extensions for multi-session assignment with observation-specific costs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pyrecest.backend import (
    __backend_name__,
)
from pyrecest.backend import all as backend_all  # pylint: disable=no-name-in-module
from pyrecest.backend import (
    asarray,
    cast,
)
from pyrecest.backend import copy as backend_copy
from pyrecest.backend import (
    float64,
    full,
    isfinite,
)
from pyrecest.backend import max as backend_max
from pyrecest.backend import sum as backend_sum

from .multisession_assignment import (
    MultiSessionAssignmentResult,
    PairwiseCostsInput,
    SessionSizesInput,
    _infer_and_validate_session_sizes,
    _normalize_pairwise_costs,
    _normalize_session_sizes,
    solve_multisession_assignment,
)

ObservationCostValue = float | Sequence[float] | Any
ObservationCostsInput = (
    Mapping[int, ObservationCostValue] | Sequence[ObservationCostValue]
)


@dataclass(frozen=True)
class _ObservationCostTransform:
    session_positions: Mapping[int, int]
    start_costs: Mapping[int, Any]
    end_costs: Mapping[int, Any]
    uniform_start_cost: float
    uniform_end_cost: float
    gap_penalty: float
    cost_threshold: float | None


def solve_multisession_assignment_with_observation_costs(  # pylint: disable=R0913,R0914
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
    """Solve multi-session assignment with per-observation start/end costs."""
    assert __backend_name__ != "jax", "Not supported on JAX backend"

    _validate_scalar_cost("start_cost", start_cost)
    _validate_scalar_cost("end_cost", end_cost)
    _validate_scalar_cost("gap_penalty", gap_penalty)
    if cost_threshold is not None:
        _validate_scalar_cost("cost_threshold", cost_threshold)

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
        float(backend_max(costs)) if _array_length(costs) else float(start_cost)
        for costs in normalized_start_costs.values()
    )
    max_end_cost = max(
        float(backend_max(costs)) if _array_length(costs) else float(end_cost)
        for costs in normalized_end_costs.values()
    )

    session_positions = {
        session_idx: position
        for position, session_idx in enumerate(sorted(session_sizes_map))
    }
    transform = _ObservationCostTransform(
        session_positions=session_positions,
        start_costs=normalized_start_costs,
        end_costs=normalized_end_costs,
        uniform_start_cost=max_start_cost,
        uniform_end_cost=max_end_cost,
        gap_penalty=float(gap_penalty),
        cost_threshold=cost_threshold,
    )
    transformed_pairwise = _transform_pairwise_costs(normalized_pairwise, transform)

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

    adjusted_edges = []
    for source, target, _ in base_result.matched_edges:
        source_session, source_detection = source
        target_session, target_detection = target
        gap = _session_gap(source_session, target_session, session_positions)
        original_cost = normalized_pairwise[(source_session, target_session)][
            source_detection, target_detection
        ]
        adjusted_edges.append(
            (
                source,
                target,
                float(original_cost) + float(gap_penalty) * gap,
            )
        )

    return MultiSessionAssignmentResult(
        tracks=base_result.tracks,
        matched_edges=adjusted_edges,
        total_cost=total_cost,
    )


def _validate_scalar_cost(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _array_length(values: Any) -> int:
    return int(values.shape[0])


def _session_gap(
    source_session: int,
    target_session: int,
    session_positions: Mapping[int, int],
) -> int:
    gap = session_positions[target_session] - session_positions[source_session] - 1
    if gap < 0:
        raise ValueError("Session indices must define a forward-in-time edge ordering.")
    return gap


def _normalize_observation_costs(
    observation_costs: ObservationCostsInput | None,
    session_sizes: Mapping[int, int],
    *,
    default_value: float,
    name: str,
) -> dict[int, Any]:
    if observation_costs is None:
        raw_entries: dict[int, ObservationCostValue] = {}
    elif isinstance(observation_costs, Mapping):
        raw_entries = {
            int(session_idx): value for session_idx, value in observation_costs.items()
        }
    else:
        raw_entries = dict(enumerate(observation_costs))

    unknown_sessions = sorted(set(raw_entries) - set(session_sizes))
    if unknown_sessions:
        raise ValueError(
            f"{name} specifies unknown sessions {unknown_sessions}; "
            f"known sessions are {sorted(session_sizes)}."
        )

    normalized: dict[int, Any] = {}
    for session_idx in sorted(session_sizes):
        session_size = int(session_sizes[session_idx])
        if session_idx not in raw_entries:
            values = full((session_size,), float(default_value), dtype=float64)
        else:
            values = _normalize_observation_cost_entry(
                raw_entries[session_idx],
                session_size,
                session_idx=session_idx,
                name=name,
            )
        if not bool(backend_all(isfinite(values))):
            raise ValueError(
                f"{name} for session {session_idx} must contain only finite values."
            )
        normalized[session_idx] = values
    return normalized


def _normalize_observation_cost_entry(
    value: ObservationCostValue,
    session_size: int,
    *,
    session_idx: int,
    name: str,
) -> Any:
    values = asarray(value, dtype=float64)
    if values.ndim == 0:
        return full((session_size,), float(values), dtype=float64)
    if values.ndim != 1:
        raise ValueError(
            f"{name} entry for session {session_idx} must be a scalar or a one-dimensional array."
        )
    if _array_length(values) != int(session_size):
        raise ValueError(
            f"{name} entry for session {session_idx} has length {_array_length(values)}, "
            f"expected {session_size}."
        )
    return cast(values, float64)


def _transform_pairwise_costs(
    pairwise_costs: Mapping[tuple[int, int], Any],
    transform: _ObservationCostTransform,
) -> dict[tuple[int, int], Any]:
    transformed: dict[tuple[int, int], Any] = {}
    for (source_session, target_session), matrix in pairwise_costs.items():
        gap = _session_gap(source_session, target_session, transform.session_positions)
        source_end = transform.end_costs[source_session][:, None]
        target_start = transform.start_costs[target_session][None, :]
        transformed_matrix = (
            asarray(matrix, dtype=float64)
            - source_end
            - target_start
            + float(transform.uniform_end_cost)
            + float(transform.uniform_start_cost)
        )

        if transform.cost_threshold is not None:
            adjusted_original = (
                asarray(matrix, dtype=float64) + float(transform.gap_penalty) * gap
            )
            transformed_matrix = backend_copy(transformed_matrix)
            transformed_matrix[adjusted_original > float(transform.cost_threshold)] = (
                math.inf
            )

        transformed[(source_session, target_session)] = transformed_matrix
    return transformed


__all__ = ["solve_multisession_assignment_with_observation_costs"]
