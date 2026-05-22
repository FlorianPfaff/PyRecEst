"""Generic fixed-lag Viterbi association for single-target tracklets."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrackletAssociationCandidate:
    """One association candidate for one scan/frame.

    Parameters
    ----------
    candidate_id : Hashable
        Identifier that is unique enough for diagnostics within a frame.
    unary_cost : float
        Per-candidate negative log-cost or any additive score to minimize.
    time_s : float, optional
        Candidate timestamp.  When omitted, frame indices are used for fixed-lag
        windows and motion costs.
    track_id : Hashable, optional
        External tracklet identifier used for switch-cost penalties.
    position : array-like, optional
        Position vector used by the default constant-velocity transition cost
        when ``config.motion_weight`` is positive.
    velocity : array-like, optional
        Velocity vector used for default motion prediction when available.
    metadata : mapping, optional
        User-defined payload copied through to results.
    """

    candidate_id: Hashable
    unary_cost: float = 0.0
    time_s: float | None = None
    track_id: Hashable | None = None
    position: Any | None = None
    velocity: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackletViterbiConfig:
    """Configuration for generic tracklet Viterbi association."""

    max_candidates_per_frame: int | None = None
    missed_detection_cost: float = 7.0
    consecutive_miss_cost: float = 1.0
    switch_cost: float = 8.0
    missing_track_id_cost: float = 1.0
    motion_weight: float = 0.0
    transition_position_std: float = 1.0
    transition_velocity_std: float | None = None
    max_speed: float | None = None
    max_speed_penalty: float = 0.0
    max_candidate_pool_per_frame: int | None = None
    max_candidates_per_track_id: int = 1

    def __post_init__(self) -> None:
        if self.max_candidates_per_frame is not None and self.max_candidates_per_frame < 1:
            raise ValueError("max_candidates_per_frame must be positive or None")
        if self.max_candidate_pool_per_frame is not None and self.max_candidate_pool_per_frame < 1:
            raise ValueError("max_candidate_pool_per_frame must be positive or None")
        if self.max_candidates_per_track_id < 0:
            raise ValueError("max_candidates_per_track_id must be nonnegative")
        for name in (
            "missed_detection_cost",
            "consecutive_miss_cost",
            "switch_cost",
            "missing_track_id_cost",
            "motion_weight",
            "max_speed_penalty",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.transition_position_std <= 0.0:
            raise ValueError("transition_position_std must be positive")
        if self.transition_velocity_std is not None and self.transition_velocity_std <= 0.0:
            raise ValueError("transition_velocity_std must be positive or None")
        if self.max_speed is not None and self.max_speed <= 0.0:
            raise ValueError("max_speed must be positive or None")


@dataclass(frozen=True)
class TrackletViterbiResult:
    """Result of a Viterbi association solve."""

    path: list[TrackletAssociationCandidate | None]
    total_cost: float
    costs_by_frame: list[np.ndarray] = field(default_factory=list)
    parent_indices_by_frame: list[np.ndarray] = field(default_factory=list)
    miss_streaks_by_frame: list[np.ndarray] = field(default_factory=list)

    @property
    def selected_candidates(self) -> list[TrackletAssociationCandidate]:
        """Return non-missed candidates in path order."""

        return [candidate for candidate in self.path if candidate is not None]

    @property
    def missed_detection_count(self) -> int:
        """Return the number of missed detections in the selected path."""

        return sum(candidate is None for candidate in self.path)


@dataclass(frozen=True)
class TrackSupport:
    """Prefix-only support statistics for a candidate track id."""

    count: int
    span_s: float
    continuity: float
    score: float


@dataclass(frozen=True)
class _Node:
    candidate: TrackletAssociationCandidate | None
    unary_cost: float

    @property
    def is_miss(self) -> bool:
        return self.candidate is None


TransitionCost = Callable[
    [TrackletAssociationCandidate | None, TrackletAssociationCandidate | None, int],
    float,
]


def solve_tracklet_viterbi(
    frames: Sequence[Sequence[TrackletAssociationCandidate]],
    *,
    config: TrackletViterbiConfig | None = None,
    transition_cost: TransitionCost | None = None,
    include_missed_detection: bool = True,
    return_tables: bool = False,
) -> TrackletViterbiResult:
    """Select a minimum-cost single-target path through candidate frames.

    The path contains one entry per input frame.  ``None`` denotes the missed
    detection branch for that frame.
    """

    config = TrackletViterbiConfig() if config is None else config
    nodes_by_frame = [_nodes_for_frame(frame, config, include_missed_detection) for frame in frames]
    if not nodes_by_frame:
        return TrackletViterbiResult([], 0.0)
    if any(not nodes for nodes in nodes_by_frame):
        raise ValueError("each frame must contain at least one candidate or allow missed detection")

    transition = transition_cost or (lambda previous, current, miss_streak: default_tracklet_transition_cost(previous, current, miss_streak, config))
    first_costs = np.array(
        [node.unary_cost + (config.missed_detection_cost if node.is_miss else 0.0) for node in nodes_by_frame[0]],
        dtype=float,
    )
    costs = [first_costs]
    miss_streaks = [np.array([1 if node.is_miss else 0 for node in nodes_by_frame[0]], dtype=int)]
    parents = [np.full(len(nodes_by_frame[0]), -1, dtype=int)]

    for frame_index in range(1, len(nodes_by_frame)):
        previous_nodes = nodes_by_frame[frame_index - 1]
        current_nodes = nodes_by_frame[frame_index]
        current_costs = np.empty(len(current_nodes), dtype=float)
        current_miss_streaks = np.empty(len(current_nodes), dtype=int)
        current_parents = np.empty(len(current_nodes), dtype=int)
        for current_index, current_node in enumerate(current_nodes):
            alternatives = np.array(
                [
                    costs[-1][previous_index]
                    + float(
                        transition(
                            previous_node.candidate,
                            current_node.candidate,
                            int(miss_streaks[-1][previous_index]),
                        )
                    )
                    for previous_index, previous_node in enumerate(previous_nodes)
                ],
                dtype=float,
            )
            parent = int(np.argmin(alternatives))
            current_parents[current_index] = parent
            current_costs[current_index] = current_node.unary_cost + alternatives[parent]
            current_miss_streaks[current_index] = (
                int(miss_streaks[-1][parent]) + 1 if current_node.is_miss else 0
            )
        costs.append(current_costs)
        miss_streaks.append(current_miss_streaks)
        parents.append(current_parents)

    terminal = int(np.argmin(costs[-1]))
    path_nodes = _reconstruct_path(nodes_by_frame, parents, terminal)
    result = TrackletViterbiResult(
        path=[node.candidate for node in path_nodes],
        total_cost=float(costs[-1][terminal]),
        costs_by_frame=costs if return_tables else [],
        parent_indices_by_frame=parents if return_tables else [],
        miss_streaks_by_frame=miss_streaks if return_tables else [],
    )
    return result


def solve_fixed_lag_tracklet_viterbi(
    frames: Sequence[Sequence[TrackletAssociationCandidate]],
    *,
    lag_s: float,
    config: TrackletViterbiConfig | None = None,
    transition_cost: TransitionCost | None = None,
    include_missed_detection: bool = True,
) -> TrackletViterbiResult:
    """Commit Viterbi decisions using at most ``lag_s`` future context.

    Each frame is solved on a local look-ahead window.  The previous non-missed
    committed candidate is prepended as a zero-cost prefix candidate, preserving
    dynamic consistency without using unbounded future information.
    """

    if lag_s <= 0.0:
        raise ValueError("lag_s must be positive")
    config = TrackletViterbiConfig() if config is None else config
    if not frames:
        return TrackletViterbiResult([], 0.0)

    frame_times = [_frame_time(frame, index) for index, frame in enumerate(frames)]
    path: list[TrackletAssociationCandidate | None] = []
    previous_committed: TrackletAssociationCandidate | None = None
    total_cost = 0.0

    for frame_index, start_time in enumerate(frame_times):
        end_time = start_time + float(lag_s)
        window_end = frame_index
        while window_end + 1 < len(frames) and frame_times[window_end + 1] <= end_time:
            window_end += 1
        window_frames = [list(frame) for frame in frames[frame_index : window_end + 1]]
        prefix_added = previous_committed is not None
        if prefix_added:
            assert previous_committed is not None
            window_frames.insert(0, [replace(previous_committed, unary_cost=0.0)])
        local = solve_tracklet_viterbi(
            window_frames,
            config=config,
            transition_cost=transition_cost,
            include_missed_detection=include_missed_detection,
        )
        selected = local.path[1 if prefix_added else 0]
        path.append(selected)
        total_cost += float(local.total_cost)
        if selected is not None:
            previous_committed = selected

    return TrackletViterbiResult(path=path, total_cost=total_cost)


def default_tracklet_transition_cost(
    previous: TrackletAssociationCandidate | None,
    current: TrackletAssociationCandidate | None,
    previous_miss_streak: int,
    config: TrackletViterbiConfig | None = None,
) -> float:
    """Default transition cost with missed detections, switches, and motion."""

    config = TrackletViterbiConfig() if config is None else config
    if current is None:
        return float(config.missed_detection_cost) + (
            float(config.consecutive_miss_cost) if previous is None else 0.0
        )
    if previous is None:
        return 0.0

    cost = _track_switch_cost(previous.track_id, current.track_id, config)
    if config.motion_weight > 0.0:
        cost += float(config.motion_weight) * _motion_cost(previous, current, config)
    return float(cost)


def retain_top_and_track_representatives(
    candidates: Sequence[TrackletAssociationCandidate],
    *,
    config: TrackletViterbiConfig | None = None,
) -> list[TrackletAssociationCandidate]:
    """Retain top unary candidates plus best representatives per track id."""

    config = TrackletViterbiConfig() if config is None else config
    if not candidates:
        return []
    ordered = sorted(enumerate(candidates), key=lambda item: (float(item[1].unary_cost), item[0]))
    top_k = config.max_candidates_per_frame or len(ordered)
    max_pool = config.max_candidate_pool_per_frame or max(top_k, top_k + 8)
    keep: set[int] = {index for index, _ in ordered[:top_k]}

    if config.max_candidates_per_track_id > 0:
        kept_by_track: dict[Hashable, int] = {}
        for index, candidate in ordered:
            if candidate.track_id is None:
                continue
            kept_count = kept_by_track.get(candidate.track_id, 0)
            if kept_count >= config.max_candidates_per_track_id:
                continue
            keep.add(index)
            kept_by_track[candidate.track_id] = kept_count + 1
            if len(keep) >= max_pool:
                break
    return [candidate for index, candidate in ordered if index in keep][:max_pool]


def prefix_track_support(
    frames: Sequence[Sequence[TrackletAssociationCandidate]],
) -> dict[int, dict[Hashable, TrackSupport]]:
    """Return prefix-only track support statistics for each frame."""

    support_by_frame: dict[int, dict[Hashable, TrackSupport]] = {}
    previous: list[TrackletAssociationCandidate] = []
    for frame_index, frame in enumerate(frames):
        support_by_frame[frame_index] = track_support_by_id(previous)
        previous.extend(candidate for candidate in frame if candidate.track_id is not None)
    return support_by_frame


def track_support_by_id(candidates: Sequence[TrackletAssociationCandidate]) -> dict[Hashable, TrackSupport]:
    """Return support statistics for finite/non-null track ids."""

    grouped: dict[Hashable, list[TrackletAssociationCandidate]] = {}
    for candidate in candidates:
        if candidate.track_id is not None:
            grouped.setdefault(candidate.track_id, []).append(candidate)
    support: dict[Hashable, TrackSupport] = {}
    for track_id, group in grouped.items():
        times = [float(candidate.time_s) for candidate in group if candidate.time_s is not None]
        span_s = max(times) - min(times) if len(times) >= 2 else 0.0
        count = len(group)
        if times:
            frame_span = max(float(count), span_s + 1.0)
        else:
            frame_span = float(count)
        continuity = float(np.clip(count / max(frame_span, 1.0), 0.0, 1.0))
        score = float(np.log1p(count) + 0.5 * np.log1p(max(span_s, 0.0)) + 0.5 * continuity)
        support[track_id] = TrackSupport(count=count, span_s=float(span_s), continuity=continuity, score=score)
    return support


def track_support_cost(
    candidate: TrackletAssociationCandidate,
    support_by_id: Mapping[Hashable, TrackSupport],
    *,
    weight: float = 0.45,
    max_reward: float = 4.0,
) -> float:
    """Return a bounded negative cost for candidates with supported track ids."""

    if candidate.track_id is None:
        return 0.0
    support = support_by_id.get(candidate.track_id)
    if support is None:
        return 0.0
    weight = max(0.0, float(weight))
    max_reward = max(0.0, float(max_reward))
    if weight <= 0.0 or max_reward <= 0.0:
        return 0.0
    return -float(min(max_reward, weight * max(0.0, support.score)))


def _nodes_for_frame(
    frame: Sequence[TrackletAssociationCandidate],
    config: TrackletViterbiConfig,
    include_missed_detection: bool,
) -> list[_Node]:
    candidates = retain_top_and_track_representatives(frame, config=config)
    nodes = [_Node(candidate, float(candidate.unary_cost)) for candidate in candidates]
    if include_missed_detection:
        nodes.append(_Node(None, 0.0))
    return nodes


def _reconstruct_path(
    nodes_by_frame: list[list[_Node]],
    parents: list[np.ndarray],
    terminal_index: int,
) -> list[_Node]:
    best = int(terminal_index)
    path: list[_Node] = []
    for frame_index in range(len(nodes_by_frame) - 1, -1, -1):
        path.append(nodes_by_frame[frame_index][best])
        best = int(parents[frame_index][best])
        if best < 0:
            break
    path.reverse()
    return path


def _track_switch_cost(
    previous_track_id: Hashable | None,
    current_track_id: Hashable | None,
    config: TrackletViterbiConfig,
) -> float:
    if previous_track_id is None:
        return 0.0
    if current_track_id is None:
        return float(config.missing_track_id_cost)
    return 0.0 if previous_track_id == current_track_id else float(config.switch_cost)


def _motion_cost(
    previous: TrackletAssociationCandidate,
    current: TrackletAssociationCandidate,
    config: TrackletViterbiConfig,
) -> float:
    if previous.position is None or current.position is None:
        return 0.0
    previous_position = np.asarray(previous.position, dtype=float).reshape(-1)
    current_position = np.asarray(current.position, dtype=float).reshape(-1)
    if previous_position.shape != current_position.shape:
        raise ValueError("candidate positions must have matching shapes")
    dt_s = max(_candidate_time(current, 1.0) - _candidate_time(previous, 0.0), 1.0e-9)
    if previous.velocity is None:
        predicted = previous_position
    else:
        predicted = previous_position + np.asarray(previous.velocity, dtype=float).reshape(previous_position.shape) * dt_s
    position_cost = float(np.sum(((current_position - predicted) / float(config.transition_position_std)) ** 2))
    speed_cost = 0.0
    displacement_velocity = (current_position - previous_position) / dt_s
    if config.max_speed is not None:
        speed_excess = max(0.0, float(np.linalg.norm(displacement_velocity)) - float(config.max_speed))
        if speed_excess > 0.0:
            speed_cost = float(config.max_speed_penalty) * speed_excess**2
    velocity_cost = 0.0
    if config.transition_velocity_std is not None and current.velocity is not None:
        velocity = np.asarray(current.velocity, dtype=float).reshape(displacement_velocity.shape)
        velocity_cost = float(np.sum(((velocity - displacement_velocity) / float(config.transition_velocity_std)) ** 2))
    return position_cost + speed_cost + velocity_cost


def _candidate_time(candidate: TrackletAssociationCandidate, fallback: float) -> float:
    return fallback if candidate.time_s is None else float(candidate.time_s)


def _frame_time(frame: Sequence[TrackletAssociationCandidate], frame_index: int) -> float:
    times = [float(candidate.time_s) for candidate in frame if candidate.time_s is not None]
    return float(np.median(times)) if times else float(frame_index)


__all__ = [
    "TrackSupport",
    "TrackletAssociationCandidate",
    "TrackletViterbiConfig",
    "TrackletViterbiResult",
    "default_tracklet_transition_cost",
    "prefix_track_support",
    "retain_top_and_track_representatives",
    "solve_fixed_lag_tracklet_viterbi",
    "solve_tracklet_viterbi",
    "track_support_by_id",
    "track_support_cost",
]
