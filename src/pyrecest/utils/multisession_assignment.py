"""Global multi-session association utilities.

This module solves a common batch-linking problem for longitudinal cell tracking:
start from pairwise ROI matching costs between imaging sessions and recover a set
of globally consistent tracks, with at most one detection per session on each
track. The formulation supports skipped sessions through optional cross-session
edges and a configurable gap penalty.

The optimization is a minimum-cost path-cover problem on a DAG. Relative to
leaving every observation disconnected, each admissible edge contributes a
*gain* of

``start_cost + end_cost - adjusted_link_cost``.

Only edges with positive gain can improve the objective, so the solver operates
on admissible candidate edges rather than on all ``N^2`` observation pairs. The
resulting sparse bipartite matching problem is solved as a rectangular linear
assignment with per-row dummy "unmatched" columns using SciPy's sparse
Jonker-Volgenant implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pyrecest.backend import (
    __backend_name__,
    arange,
    array,
    asarray,
    cast,
    concatenate,
    empty,
    float64,
    full,
    int64,
    isfinite,
)
from pyrecest.backend import min as backend_min  # pylint: disable=no-name-in-module
from pyrecest.backend import (
    ones,
    where,
)
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

Observation = tuple[int, int]
MatchedEdge = tuple[Observation, Observation, float]
PairwiseCostsInput = Mapping[tuple[int, int], Any] | Sequence[Any]
SessionSizesInput = Mapping[int, int] | Sequence[int]
TrackInput = Mapping[int, int] | Sequence[Observation]


def _full_1d(size: int, fill_value: int | float, dtype: Any) -> Any:
    return full((int(size),), fill_value, dtype=dtype)


def _false_mask(size: int) -> Any:
    size = int(size)
    return array([False] * max(size, 1))[:size]


@dataclass(frozen=True)
class MultiSessionAssignmentResult:
    """Result of :func:`solve_multisession_assignment`.

    Attributes
    ----------
    tracks
        One dictionary per recovered track. Keys are session indices and values
        are detection indices within that session.
    matched_edges
        Directed edges selected by the optimizer as
        ``((source_session, source_detection), (target_session, target_detection), adjusted_cost)``.
        The reported edge cost already includes the configured gap penalty.
    total_cost
        Objective value of the globally optimal assignment, including track
        start and end costs.
    """

    tracks: list[dict[int, int]]
    matched_edges: list[MatchedEdge]
    total_cost: float

    def observation_to_track_index(self) -> dict[Observation, int]:
        """Return a mapping from observation to recovered track index."""
        mapping: dict[Observation, int] = {}
        for track_idx, track in enumerate(self.tracks):
            for session_idx, detection_idx in track.items():
                mapping[(session_idx, detection_idx)] = track_idx
        return mapping

    def to_session_labels(
        self,
        session_sizes: SessionSizesInput | None = None,
        *,
        fill_value: int = -1,
    ) -> tuple[Any, ...]:
        """Convert the recovered tracks to dense per-session label arrays.

        Parameters
        ----------
        session_sizes
            Optional per-session detection counts. When omitted, sizes are
            inferred from the track content.
        fill_value
            Value used for unassigned detections.
        """

        return tracks_to_session_labels(
            self.tracks,
            session_sizes=session_sizes,
            fill_value=fill_value,
        )


def solve_multisession_assignment(  # pylint: disable=too-many-locals
    pairwise_costs: PairwiseCostsInput,
    session_sizes: SessionSizesInput | None = None,
    *,
    start_cost: float = 0.0,
    end_cost: float = 0.0,
    gap_penalty: float = 0.0,
    cost_threshold: float | None = None,
) -> MultiSessionAssignmentResult:
    """Recover globally consistent tracks across multiple sessions.

    Parameters
    ----------
    pairwise_costs
        Either a mapping from ``(source_session, target_session)`` to a cost
        matrix of shape ``(n_source, n_target)``, or a sequence of consecutive
        session-to-session cost matrices. When a sequence is supplied, matrix
        ``k`` is interpreted as the cost matrix for session ``k`` to session
        ``k + 1``.
    session_sizes
        Optional per-session detection counts. This is only required when some
        sessions have no pairwise cost matrix, or when isolated singleton
        detections should still be represented in the result. A sequence is
        interpreted as counts for sessions ``0, 1, ..., S-1``.
    start_cost
        Penalty for starting a new track.
    end_cost
        Penalty for ending a track.
    gap_penalty
        Additional penalty applied per skipped session for non-consecutive
        edges. This penalty is added on top of the supplied edge cost.
    cost_threshold
        Optional upper bound for admissible link costs after gap penalties are
        applied. Edges with larger costs are forbidden.

    Returns
    -------
    MultiSessionAssignmentResult
        Globally optimal tracks and the selected directed edges.

    Notes
    -----
    Let ``N`` be the total number of observations. If every observation starts
    as a singleton track, the baseline cost is

    ``N * (start_cost + end_cost)``.

    Adding an admissible edge with adjusted cost ``c`` merges two track ends and
    therefore changes the objective by ``c - start_cost - end_cost``. The
    implementation maximizes the total gain

    ``start_cost + end_cost - c``

    subject to at-most-one-predecessor and at-most-one-successor constraints.
    This is equivalent to the minimum-cost path-cover objective. Internally,
    the sparse matching problem is reduced to a rectangular assignment in which
    every source observation can either connect to an admissible successor or to
    its own dummy "unmatched" column.
    """

    assert __backend_name__ != "jax", "Not supported on JAX backend"

    _validate_scalar_cost("start_cost", start_cost)
    _validate_scalar_cost("end_cost", end_cost)
    _validate_scalar_cost("gap_penalty", gap_penalty)
    if cost_threshold is not None:
        _validate_scalar_cost("cost_threshold", cost_threshold)

    normalized_pairwise_costs = _normalize_pairwise_costs(pairwise_costs)
    normalized_session_sizes = _normalize_session_sizes(session_sizes)
    session_sizes_map = _infer_and_validate_session_sizes(
        normalized_pairwise_costs,
        normalized_session_sizes,
    )

    if not session_sizes_map:
        return MultiSessionAssignmentResult(tracks=[], matched_edges=[], total_cost=0.0)

    session_order = sorted(session_sizes_map)
    session_positions = {
        session_idx: position for position, session_idx in enumerate(session_order)
    }
    session_offsets = _build_session_offsets(session_sizes_map)
    global_to_observation, _ = _build_observation_index(session_sizes_map)
    n_observations = len(global_to_observation)

    if n_observations == 0:
        return MultiSessionAssignmentResult(tracks=[], matched_edges=[], total_cost=0.0)

    left_nodes, right_nodes, edge_gains, adjusted_costs = _build_candidate_edges(
        normalized_pairwise_costs,
        session_sizes_map,
        session_positions,
        session_offsets,
        start_cost=start_cost,
        end_cost=end_cost,
        gap_penalty=gap_penalty,
        cost_threshold=cost_threshold,
    )

    selected_edge_mask = _solve_max_weight_matching(
        left_nodes,
        right_nodes,
        edge_gains,
        n_observations,
    )

    predecessors = _full_1d(n_observations, -1, int64)
    successors = _full_1d(n_observations, -1, int64)
    matched_edges: list[MatchedEdge] = []

    for edge_index in where(selected_edge_mask)[0]:
        source_global = int(left_nodes[edge_index])
        target_global = int(right_nodes[edge_index])

        if successors[source_global] != -1 or predecessors[target_global] != -1:
            raise RuntimeError("The selected edges do not define a valid matching.")

        successors[source_global] = target_global
        predecessors[target_global] = source_global
        matched_edges.append(
            (
                global_to_observation[source_global],
                global_to_observation[target_global],
                float(adjusted_costs[edge_index]),
            )
        )

    tracks = _reconstruct_tracks(
        global_to_observation,
        predecessors,
        successors,
        session_positions,
    )

    matched_edges.sort(
        key=lambda edge: (
            session_positions[edge[0][0]],
            edge[0][1],
            session_positions[edge[1][0]],
            edge[1][1],
        )
    )

    baseline_cost = float(n_observations) * (float(start_cost) + float(end_cost))
    total_cost = baseline_cost - float(edge_gains[selected_edge_mask].sum())

    # Basic internal consistency check.
    observation_count = sum(len(track) for track in tracks)
    if observation_count != n_observations:
        raise RuntimeError("Failed to reconstruct a full observation cover.")

    # Ensure every selected edge connects observations that share a track.
    observation_to_track = {}
    for track_index, track in enumerate(tracks):
        for session_index, detection_index in track.items():
            observation_to_track[(session_index, detection_index)] = track_index
    for edge in matched_edges:
        if observation_to_track[edge[0]] != observation_to_track[edge[1]]:
            raise RuntimeError(
                "Recovered edges are inconsistent with reconstructed tracks."
            )

    return MultiSessionAssignmentResult(
        tracks=tracks,
        matched_edges=matched_edges,
        total_cost=total_cost,
    )


def tracks_to_session_labels(
    tracks: Sequence[TrackInput],
    session_sizes: SessionSizesInput | None = None,
    *,
    fill_value: int = -1,
) -> tuple[Any, ...]:
    """Convert explicit tracks to dense per-session label arrays.

    Parameters
    ----------
    tracks
        Sequence of track representations. Each track can either be a mapping
        from session index to detection index or a sequence of
        ``(session_index, detection_index)`` pairs.
    session_sizes
        Optional per-session detection counts. When omitted, sizes are inferred
        from the maximum detection index present in each session. Missing
        sessions are represented by empty arrays.
    fill_value
        Value used for detections that are not assigned to any track.

    Returns
    -------
    tuple
        One integer array per session, indexed by session number.
    """

    assert __backend_name__ != "jax", "Not supported on JAX backend"

    inferred_sizes, max_session_index = _validate_track_session_sizes(
        tracks,
        session_sizes,
    )

    labels = [
        _full_1d(inferred_sizes.get(session_index, 0), fill_value, int64)
        for session_index in range(max_session_index + 1)
    ]

    for track_index, track in enumerate(tracks):
        for session_index, detection_index in _iter_track_items(track):
            if labels[session_index][detection_index] != fill_value:
                raise ValueError("Each detection can only belong to a single track.")
            labels[session_index][detection_index] = track_index

    return tuple(labels)


def _validate_track_session_sizes(
    tracks: Sequence[TrackInput],
    session_sizes: SessionSizesInput | None = None,
    *,
    require_unique_sessions: bool = False,
) -> tuple[dict[int, int], int]:
    inferred_sizes = _normalize_session_sizes(session_sizes)
    max_session_index = max(inferred_sizes, default=-1)

    for track in tracks:
        seen_sessions: set[int] = set()
        for session_index, detection_index in _iter_track_items(track):
            if detection_index < 0:
                raise ValueError(
                    f"Detection indices must be non-negative, got {detection_index}."
                )
            if require_unique_sessions:
                if session_index in seen_sessions:
                    raise ValueError(
                        "Each track can only contain one detection per session."
                    )
                seen_sessions.add(session_index)
            session_index = int(session_index)
            candidate_size = int(detection_index) + 1
            max_session_index = max(max_session_index, session_index)
            current_size = inferred_sizes.get(session_index, 0)
            if session_sizes is None:
                inferred_sizes[session_index] = max(current_size, candidate_size)
            elif candidate_size > current_size:
                raise ValueError(
                    f"Track references detection {detection_index} in session {session_index}, "
                    f"but session_sizes only allows {current_size} detections."
                )

    return inferred_sizes, max_session_index


def _validate_scalar_cost(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _normalize_pairwise_costs(
    pairwise_costs: PairwiseCostsInput,
) -> dict[tuple[int, int], Any]:
    if isinstance(pairwise_costs, Mapping):
        normalized: dict[tuple[int, int], Any] = {}
        for key, value in pairwise_costs.items():
            if len(key) != 2:
                raise ValueError(
                    "Each pairwise-cost key must contain two session indices."
                )
            source_session, target_session = int(key[0]), int(key[1])
            if source_session >= target_session:
                raise ValueError(
                    "Pairwise-cost keys must satisfy source_session < target_session."
                )
            matrix = asarray(value, dtype=float64)
            if matrix.ndim != 2:
                raise ValueError("Each pairwise cost matrix must be two-dimensional.")
            normalized[(source_session, target_session)] = matrix
        return normalized

    normalized = {}
    for session_idx, value in enumerate(pairwise_costs):
        matrix = asarray(value, dtype=float64)
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
        normalized = {
            int(session_idx): int(size) for session_idx, size in session_sizes.items()
        }
    else:
        normalized = {
            session_idx: int(size) for session_idx, size in enumerate(session_sizes)
        }
    for session_idx, size in normalized.items():
        if size < 0:
            raise ValueError(f"Session {session_idx} has a negative detection count.")
    return normalized


def _infer_and_validate_session_sizes(
    pairwise_costs: Mapping[tuple[int, int], Any],
    session_sizes: Mapping[int, int],
) -> dict[int, int]:
    inferred_sizes = dict(session_sizes)
    for (source_session, target_session), cost_matrix in pairwise_costs.items():
        source_size, target_size = cost_matrix.shape
        _check_or_set_session_size(inferred_sizes, source_session, source_size)
        _check_or_set_session_size(inferred_sizes, target_session, target_size)

    if not inferred_sizes and not pairwise_costs:
        raise ValueError(
            "No observations were provided. Supply pairwise_costs or session_sizes."
        )

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


def _build_session_offsets(session_sizes: Mapping[int, int]) -> dict[int, int]:
    offsets: dict[int, int] = {}
    offset = 0
    for session_idx in sorted(session_sizes):
        offsets[session_idx] = offset
        offset += int(session_sizes[session_idx])
    return offsets


def _build_observation_index(
    session_sizes: Mapping[int, int],
) -> tuple[list[Observation], dict[Observation, int]]:
    global_to_observation: list[Observation] = []
    observation_to_global: dict[Observation, int] = {}

    for session_idx in sorted(session_sizes):
        for detection_idx in range(session_sizes[session_idx]):
            observation = (session_idx, detection_idx)
            observation_to_global[observation] = len(global_to_observation)
            global_to_observation.append(observation)

    return global_to_observation, observation_to_global


def _build_candidate_edges(  # pylint: disable=too-many-arguments,too-many-locals
    pairwise_costs: Mapping[tuple[int, int], Any],
    session_sizes: Mapping[int, int],
    session_positions: Mapping[int, int],
    session_offsets: Mapping[int, int],
    *,
    start_cost: float,
    end_cost: float,
    gap_penalty: float,
    cost_threshold: float | None,
) -> tuple[Any, Any, Any, Any]:
    left_nodes: list[Any] = []
    right_nodes: list[Any] = []
    edge_gains: list[Any] = []
    adjusted_costs: list[Any] = []

    for (source_session, target_session), cost_matrix in sorted(pairwise_costs.items()):
        expected_shape = (session_sizes[source_session], session_sizes[target_session])
        if cost_matrix.shape != expected_shape:
            raise ValueError(
                f"Pairwise matrix {(source_session, target_session)} has shape "
                f"{cost_matrix.shape}, expected {expected_shape}."
            )

        source_position = session_positions[source_session]
        target_position = session_positions[target_session]
        gap = target_position - source_position - 1
        if gap < 0:
            raise ValueError(
                "Session indices must define a forward-in-time edge ordering."
            )

        adjusted_matrix = asarray(cost_matrix, dtype=float64) + float(gap_penalty) * gap
        gain_matrix = (float(start_cost) + float(end_cost)) - adjusted_matrix

        valid_mask = isfinite(adjusted_matrix) & (gain_matrix > 0.0)
        if cost_threshold is not None:
            valid_mask &= adjusted_matrix <= float(cost_threshold)

        source_indices, target_indices = where(valid_mask)
        if source_indices.shape[0] == 0:
            continue

        left_nodes.append(session_offsets[source_session] + cast(source_indices, int64))
        right_nodes.append(
            session_offsets[target_session] + cast(target_indices, int64)
        )
        edge_gains.append(cast(gain_matrix[valid_mask], float64))
        adjusted_costs.append(cast(adjusted_matrix[valid_mask], float64))

    if not edge_gains:
        return (
            empty((0,), dtype=int64),
            empty((0,), dtype=int64),
            empty((0,), dtype=float64),
            empty((0,), dtype=float64),
        )

    return (
        concatenate(left_nodes),
        concatenate(right_nodes),
        concatenate(edge_gains),
        concatenate(adjusted_costs),
    )


def _solve_max_weight_matching(
    left_nodes: Any,
    right_nodes: Any,
    edge_gains: Any,
    num_nodes: int,
) -> Any:
    try:
        return _solve_max_weight_matching_sparse(
            left_nodes,
            right_nodes,
            edge_gains,
            num_nodes,
        )
    except ValueError:
        return _solve_max_weight_matching_via_linprog(
            left_nodes,
            right_nodes,
            edge_gains,
            num_nodes,
        )


def _solve_max_weight_matching_sparse(  # pylint: disable=too-many-locals
    left_nodes: Any,
    right_nodes: Any,
    edge_gains: Any,
    num_nodes: int,
) -> Any:
    """Solve the sparse bipartite matching problem with LAPJVsp.

    The optional-matching objective is turned into a full rectangular
    assignment: every left node is matched either to one admissible right node
    or to its own dummy "unmatched" column. Because the sparse assignment
    always covers all rows, adding the same positive offset to every row edge
    preserves the original maximum-gain objective while satisfying SciPy's
    requirement that sparse edge weights are non-zero.
    """

    num_edges = int(edge_gains.shape[0])
    if num_edges == 0 or num_nodes == 0:
        return _false_mask(num_edges)

    left_nodes = asarray(left_nodes, dtype=int64)
    right_nodes = asarray(right_nodes, dtype=int64)
    edge_gains = asarray(edge_gains, dtype=float64)
    node_ids = arange(num_nodes, dtype=int64)

    row_ids = concatenate((left_nodes, node_ids))
    col_ids = concatenate((right_nodes, num_nodes + node_ids))

    base_weight = min(1.0, 0.5 * float(backend_min(edge_gains)))
    base_weight = max(base_weight, math.ulp(0.0))
    weights = concatenate(
        (
            edge_gains + base_weight,
            _full_1d(num_nodes, base_weight, float64),
        )
    )

    biadjacency = coo_matrix(
        (weights, (row_ids, col_ids)),
        shape=(num_nodes, 2 * num_nodes),
    ).tocsr()

    matched_rows, matched_cols = min_weight_full_bipartite_matching(
        biadjacency,
        maximize=True,
    )

    if matched_rows.size != num_nodes:
        raise RuntimeError("Sparse assignment did not produce a full row cover.")

    edge_lookup = {
        (int(source_node), int(target_node)): edge_index
        for edge_index, (source_node, target_node) in enumerate(
            zip(left_nodes, right_nodes)
        )
    }
    selected_edge_mask = _false_mask(num_edges)

    for source_node, assigned_column in zip(matched_rows, matched_cols):
        assigned_column = int(assigned_column)
        if assigned_column >= num_nodes:
            continue
        edge_index = edge_lookup.get((int(source_node), assigned_column))
        if edge_index is None:
            raise RuntimeError("Sparse assignment selected a non-admissible edge.")
        selected_edge_mask[edge_index] = True

    return selected_edge_mask


def _solve_max_weight_matching_via_linprog(
    left_nodes: Any,
    right_nodes: Any,
    edge_gains: Any,
    num_nodes: int,
) -> Any:
    """Fallback solver that keeps the original LP formulation."""

    num_edges = int(edge_gains.shape[0])
    if num_edges == 0 or num_nodes == 0:
        return _false_mask(num_edges)

    edge_ids = arange(num_edges, dtype=int64)
    row_ids = concatenate((left_nodes, num_nodes + right_nodes))
    col_ids = concatenate((edge_ids, edge_ids))

    constraint_matrix = coo_matrix(
        (ones((2 * num_edges,), dtype=float64), (row_ids, col_ids)),
        shape=(2 * num_nodes, num_edges),
    ).tocsr()

    solution = linprog(
        c=-edge_gains,
        A_ub=constraint_matrix,
        b_ub=ones((2 * num_nodes,), dtype=float64),
        bounds=(0.0, 1.0),
        method="highs",
    )

    if not solution.success:
        raise RuntimeError(
            "Global multi-session association failed: " f"{solution.message}"
        )

    return array(solution.x > 0.5)


def _reconstruct_tracks(
    global_to_observation: Sequence[Observation],
    predecessors: Any,
    successors: Any,
    session_positions: Mapping[int, int],
) -> list[dict[int, int]]:
    visited: set[int] = set()
    tracks: list[dict[int, int]] = []

    for node_idx in range(len(global_to_observation)):
        if predecessors[node_idx] != -1 or node_idx in visited:
            continue

        track: dict[int, int] = {}
        current_node = node_idx
        while current_node != -1:
            if current_node in visited:
                raise RuntimeError("Encountered a cycle while reconstructing tracks.")
            visited.add(current_node)
            session_idx, detection_idx = global_to_observation[current_node]
            track[session_idx] = detection_idx
            current_node = int(successors[current_node])
        tracks.append(track)

    for node_idx in range(len(global_to_observation)):
        if node_idx in visited:
            continue
        session_idx, detection_idx = global_to_observation[node_idx]
        tracks.append({session_idx: detection_idx})
        visited.add(node_idx)

    tracks.sort(
        key=lambda track: (
            min(session_positions[session_idx] for session_idx in track),
            min(track[session_idx] for session_idx in track),
            len(track),
        )
    )
    return tracks


def _iter_track_items(track: TrackInput) -> list[Observation]:
    if isinstance(track, Mapping):
        items = list(track.items())
    else:
        items = [
            (int(session_idx), int(detection_idx))
            for session_idx, detection_idx in track
        ]
    items.sort(key=lambda item: item[0])
    return [
        (int(session_idx), int(detection_idx)) for session_idx, detection_idx in items
    ]


__all__ = [
    "MultiSessionAssignmentResult",
    "solve_multisession_assignment",
    "tracks_to_session_labels",
]
