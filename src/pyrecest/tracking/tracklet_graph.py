"""Generic tracklet DAG and k-best association path utilities.

This module is intentionally application-independent.  A caller supplies
tracklet nodes with start/end times, endpoint states, and optional metadata;
PyRecEst builds a directed acyclic graph over plausible transitions and returns
low-cost tracklet paths.  Domain-specific features such as radar class
probabilities, RF contradiction scores, or sensor-specific range gates should be
converted into node/edge costs by the calling project.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Hashable

import numpy as np

CostFn = Callable[["Tracklet"], float]
EdgeCostFn = Callable[["Tracklet", "Tracklet"], float | None]


@dataclass(frozen=True)
class Tracklet:
    """A time-bounded association primitive used as a graph node."""

    id: Hashable
    start_time: float
    end_time: float
    start_state: Any
    end_state: Any
    cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        start_time = _finite_float(self.start_time, "start_time")
        end_time = _finite_float(self.end_time, "end_time")
        if end_time < start_time:
            raise ValueError("tracklet end_time must be >= start_time")
        start_state = _state_vector(self.start_state, "start_state")
        end_state = _state_vector(self.end_state, "end_state")
        if start_state.shape != end_state.shape:
            raise ValueError("start_state and end_state must have the same shape")
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "end_time", end_time)
        object.__setattr__(self, "start_state", start_state)
        object.__setattr__(self, "end_state", end_state)
        object.__setattr__(self, "cost", _finite_float(self.cost, "cost"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def duration(self) -> float:
        """Return ``end_time - start_time``."""

        return float(self.end_time - self.start_time)


@dataclass(frozen=True)
class TrackletEdge:
    """Directed transition edge between two tracklets."""

    source_id: Hashable
    target_id: Hashable
    cost: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cost", _finite_float(self.cost, "cost"))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TrackletPath:
    """A candidate sequence of tracklets through a DAG."""

    tracklet_ids: tuple[Hashable, ...]
    cost: float
    node_cost: float = 0.0
    edge_cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tracklet_ids:
            raise ValueError("tracklet path must contain at least one tracklet id")
        cost = _finite_float(self.cost, "cost")
        node_cost = _finite_float(self.node_cost, "node_cost")
        edge_cost = _finite_float(self.edge_cost, "edge_cost")
        object.__setattr__(self, "tracklet_ids", tuple(self.tracklet_ids))
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "node_cost", node_cost)
        object.__setattr__(self, "edge_cost", edge_cost)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def length(self) -> int:
        """Return the number of tracklets in the path."""

        return int(len(self.tracklet_ids))


@dataclass(frozen=True)
class TrackletGraphConfig:
    """Configuration for k-best path extraction from a tracklet DAG."""

    top_k: int = 10
    beam_width: int | None = None
    allow_overlap: bool = False
    max_gap: float | None = None
    diversity_weight: float = 0.0
    candidate_multiplier: int = 5

    def __post_init__(self) -> None:
        top_k = _positive_integer(self.top_k, "top_k", "top_k must be positive")
        beam_width = (
            None
            if self.beam_width is None
            else _positive_integer(
                self.beam_width,
                "beam_width",
                "beam_width must be positive when supplied",
            )
        )
        max_gap = (
            None
            if self.max_gap is None
            else _nonnegative_finite_float(
                self.max_gap,
                "max_gap",
                "max_gap must be nonnegative when supplied",
            )
        )
        diversity_weight = _nonnegative_finite_float(
            self.diversity_weight,
            "diversity_weight",
            "diversity_weight must be nonnegative",
        )
        candidate_multiplier = _positive_integer(
            self.candidate_multiplier,
            "candidate_multiplier",
            "candidate_multiplier must be positive",
        )
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "beam_width", beam_width)
        object.__setattr__(self, "max_gap", max_gap)
        object.__setattr__(self, "diversity_weight", diversity_weight)
        object.__setattr__(self, "candidate_multiplier", candidate_multiplier)


def constant_velocity_edge_cost(
    *,
    max_gap: float | None = None,
    max_speed: float | None = None,
    state_slice: slice | Sequence[int] = slice(None),
    gap_weight: float = 0.0,
    speed_weight: float = 1.0,
    switch_metadata_key: str | None = None,
    switch_penalty: float = 0.0,
) -> EdgeCostFn:
    """Build a generic constant-velocity feasibility/cost function."""

    max_gap_value = (
        None
        if max_gap is None
        else _nonnegative_finite_float(
            max_gap,
            "max_gap",
            "max_gap must be nonnegative when supplied",
        )
    )
    max_speed_value = (
        None
        if max_speed is None
        else _positive_finite_float(
            max_speed,
            "max_speed",
            "max_speed must be positive when supplied",
        )
    )
    gap_weight_value = _finite_float(gap_weight, "gap_weight")
    speed_weight_value = _finite_float(speed_weight, "speed_weight")
    switch_penalty_value = _finite_float(switch_penalty, "switch_penalty")

    def edge_cost(left: Tracklet, right: Tracklet) -> float:
        gap = float(right.start_time - left.end_time)
        if gap < -1.0e-9:
            return float("inf")
        if max_gap_value is not None and gap > max_gap_value:
            return float("inf")
        dt = max(gap, 1.0e-9)
        left_state = left.end_state[state_slice]
        right_state = right.start_state[state_slice]
        distance = float(
            np.linalg.norm(np.asarray(right_state) - np.asarray(left_state))
        )
        speed = distance / dt
        if max_speed_value is not None and speed > max_speed_value:
            return float("inf")
        switch = 0.0
        if switch_metadata_key is not None:
            if left.metadata.get(switch_metadata_key) != right.metadata.get(
                switch_metadata_key
            ):
                switch = switch_penalty_value
        return float(gap_weight_value * gap + speed_weight_value * speed + switch)

    return edge_cost


def build_tracklet_adjacency(
    tracklets: Sequence[Tracklet],
    edge_cost_fn: EdgeCostFn,
    *,
    allow_overlap: bool = False,
    max_gap: float | None = None,
) -> dict[Hashable, list[tuple[Hashable, float]]]:
    """Build adjacency lists for a time-ordered tracklet DAG."""

    ordered = sort_tracklets(tracklets)
    _require_unique_tracklet_ids(ordered)
    max_gap_value = (
        None
        if max_gap is None
        else _nonnegative_finite_float(
            max_gap,
            "max_gap",
            "max_gap must be nonnegative when supplied",
        )
    )
    adjacency: dict[Hashable, list[tuple[Hashable, float]]] = {
        item.id: [] for item in ordered
    }
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            gap = float(right.start_time - left.end_time)
            if not allow_overlap and gap < -1.0e-9:
                continue
            if max_gap_value is not None and gap > max_gap_value:
                break
            cost = edge_cost_fn(left, right)
            if cost is None or not np.isfinite(float(cost)):
                continue
            adjacency[left.id].append((right.id, float(cost)))
    return adjacency


def k_best_tracklet_paths(
    tracklets: Sequence[Tracklet],
    *,
    edge_cost_fn: EdgeCostFn,
    config: TrackletGraphConfig | None = None,
    node_cost_fn: CostFn | None = None,
    start_cost_fn: CostFn | None = None,
    end_cost_fn: CostFn | None = None,
) -> list[TrackletPath]:
    """Return the k lowest-cost paths through a tracklet DAG."""

    cfg = TrackletGraphConfig() if config is None else config
    ordered = sort_tracklets(tracklets)
    if not ordered:
        return []
    adjacency = build_tracklet_adjacency(
        ordered,
        edge_cost_fn,
        allow_overlap=cfg.allow_overlap,
        max_gap=cfg.max_gap,
    )
    by_id = {item.id: item for item in ordered}
    index_by_id = {item.id: index for index, item in enumerate(ordered)}
    beam_width = cfg.beam_width or max(cfg.top_k, 1)
    paths_by_end: dict[Hashable, list[TrackletPath]] = {}

    for tracklet in ordered:
        node_cost = _node_cost(tracklet, node_cost_fn)
        start_cost = (
            0.0 if start_cost_fn is None else _finite_cost(start_cost_fn(tracklet))
        )
        candidates = [
            TrackletPath(
                tracklet_ids=(tracklet.id,),
                cost=start_cost + node_cost,
                node_cost=node_cost,
                edge_cost=0.0,
            )
        ]
        for left in ordered[: index_by_id[tracklet.id]]:
            for right_id, edge_cost in adjacency[left.id]:
                if right_id != tracklet.id:
                    continue
                for previous in paths_by_end.get(left.id, []):
                    candidates.append(
                        TrackletPath(
                            tracklet_ids=(*previous.tracklet_ids, tracklet.id),
                            cost=previous.cost + edge_cost + node_cost,
                            node_cost=previous.node_cost + node_cost,
                            edge_cost=previous.edge_cost + edge_cost,
                        )
                    )
        paths_by_end[tracklet.id] = sorted(candidates, key=_path_sort_key)[:beam_width]

    all_paths: dict[tuple[Hashable, ...], TrackletPath] = {}
    for path_list in paths_by_end.values():
        for path in path_list:
            last = by_id[path.tracklet_ids[-1]]
            end_cost = 0.0 if end_cost_fn is None else _finite_cost(end_cost_fn(last))
            final_path = TrackletPath(
                tracklet_ids=path.tracklet_ids,
                cost=path.cost + end_cost,
                node_cost=path.node_cost,
                edge_cost=path.edge_cost,
            )
            if (
                final_path.tracklet_ids not in all_paths
                or final_path.cost < all_paths[final_path.tracklet_ids].cost
            ):
                all_paths[final_path.tracklet_ids] = final_path

    return sorted(all_paths.values(), key=_path_sort_key)[: cfg.top_k]


def diverse_k_best_tracklet_paths(
    tracklets: Sequence[Tracklet],
    *,
    edge_cost_fn: EdgeCostFn,
    config: TrackletGraphConfig | None = None,
    node_cost_fn: CostFn | None = None,
    start_cost_fn: CostFn | None = None,
    end_cost_fn: CostFn | None = None,
) -> list[TrackletPath]:
    """Return k paths with an optional greedy Jaccard overlap penalty."""

    cfg = TrackletGraphConfig() if config is None else config
    if cfg.diversity_weight <= 0.0:
        return k_best_tracklet_paths(
            tracklets,
            edge_cost_fn=edge_cost_fn,
            config=cfg,
            node_cost_fn=node_cost_fn,
            start_cost_fn=start_cost_fn,
            end_cost_fn=end_cost_fn,
        )
    candidate_cfg = TrackletGraphConfig(
        top_k=max(cfg.top_k * cfg.candidate_multiplier, cfg.top_k),
        beam_width=cfg.beam_width,
        allow_overlap=cfg.allow_overlap,
        max_gap=cfg.max_gap,
        diversity_weight=0.0,
        candidate_multiplier=cfg.candidate_multiplier,
    )
    candidates = k_best_tracklet_paths(
        tracklets,
        edge_cost_fn=edge_cost_fn,
        config=candidate_cfg,
        node_cost_fn=node_cost_fn,
        start_cost_fn=start_cost_fn,
        end_cost_fn=end_cost_fn,
    )
    selected: list[TrackletPath] = []
    remaining = list(candidates)
    while remaining and len(selected) < cfg.top_k:

        def adjusted(path: TrackletPath) -> tuple[float, float, tuple[str, ...]]:
            overlap = max((path_jaccard(path, kept) for kept in selected), default=0.0)
            return (
                path.cost + cfg.diversity_weight * overlap,
                path.cost,
                tuple(str(item) for item in path.tracklet_ids),
            )

        best = min(remaining, key=adjusted)
        overlap = max((path_jaccard(best, kept) for kept in selected), default=0.0)
        selected.append(
            TrackletPath(
                tracklet_ids=best.tracklet_ids,
                cost=best.cost,
                node_cost=best.node_cost,
                edge_cost=best.edge_cost,
                metadata={**dict(best.metadata), "jaccard_to_previous": overlap},
            )
        )
        remaining = [
            path for path in remaining if path.tracklet_ids != best.tracklet_ids
        ]
    return selected


def path_jaccard(left: TrackletPath, right: TrackletPath) -> float:
    """Return Jaccard overlap between two tracklet paths."""

    left_set = set(left.tracklet_ids)
    right_set = set(right.tracklet_ids)
    union = left_set | right_set
    if not union:
        return 0.0
    return float(len(left_set & right_set) / len(union))


def materialize_tracklet_path(
    path: TrackletPath, tracklets: Mapping[Hashable, Tracklet]
) -> list[Tracklet]:
    """Return path tracklets in path order."""

    return [tracklets[tracklet_id] for tracklet_id in path.tracklet_ids]


def tracklet_paths_to_dicts(
    paths: Iterable[TrackletPath],
    *,
    tracklets: Mapping[Hashable, Tracklet] | None = None,
) -> list[dict[str, Any]]:
    """Serialize tracklet paths for CSV/JSON diagnostics."""

    rows: list[dict[str, Any]] = []
    for rank, path in enumerate(paths):
        row: dict[str, Any] = {
            "rank": int(rank),
            "tracklet_ids": ";".join(str(item) for item in path.tracklet_ids),
            "path_length": int(path.length),
            "cost": float(path.cost),
            "node_cost": float(path.node_cost),
            "edge_cost": float(path.edge_cost),
        }
        row.update(
            {f"metadata_{key}": value for key, value in dict(path.metadata).items()}
        )
        if tracklets is not None:
            members = materialize_tracklet_path(path, tracklets)
            start = min(member.start_time for member in members)
            end = max(member.end_time for member in members)
            row.update({"start_time": start, "end_time": end, "duration": end - start})
        rows.append(row)
    return rows


def sort_tracklets(tracklets: Iterable[Tracklet]) -> list[Tracklet]:
    """Return tracklets sorted by start time, end time, and id string."""

    return sorted(
        tracklets, key=lambda item: (item.start_time, item.end_time, str(item.id))
    )


def _require_unique_tracklet_ids(tracklets: Sequence[Tracklet]) -> None:
    seen: set[Hashable] = set()
    duplicates: set[str] = set()
    for item in tracklets:
        try:
            if item.id in seen:
                duplicates.add(str(item.id))
            else:
                seen.add(item.id)
        except TypeError as exc:
            raise ValueError("tracklet ids must be hashable") from exc
    if duplicates:
        duplicate_text = ", ".join(sorted(duplicates))
        raise ValueError(
            f"tracklet ids must be unique; duplicate id(s): {duplicate_text}"
        )


def _node_cost(tracklet: Tracklet, node_cost_fn: CostFn | None) -> float:
    if node_cost_fn is None:
        return float(tracklet.cost)
    return _finite_cost(node_cost_fn(tracklet))


def _finite_cost(value: float) -> float:
    value = _finite_float(value, "cost")
    if not np.isfinite(value):
        raise ValueError("cost functions must return finite costs")
    return value


def _path_sort_key(path: TrackletPath) -> tuple[float, int, tuple[str, ...]]:
    return (
        float(path.cost),
        int(path.length),
        tuple(str(item) for item in path.tracklet_ids),
    )


def _state_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def _positive_integer(value: Any, name: str, message: str) -> int:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(message)
    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(scalar, (int, np.integer)):
        parsed = int(scalar)
    else:
        try:
            scalar_float = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if not np.isfinite(scalar_float) or not scalar_float.is_integer():
            raise ValueError(message)
        parsed = int(scalar_float)
    if parsed <= 0:
        raise ValueError(message)
    return parsed


def _nonnegative_finite_float(value: Any, name: str, message: str) -> float:
    parsed = _finite_float(value, name)
    if parsed < 0.0:
        raise ValueError(message)
    return parsed


def _positive_finite_float(value: Any, name: str, message: str) -> float:
    parsed = _finite_float(value, name)
    if parsed <= 0.0:
        raise ValueError(message)
    return parsed


def _finite_float(value: Any, name: str) -> float:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be finite")
    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        parsed = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


__all__ = [
    "CostFn",
    "EdgeCostFn",
    "Tracklet",
    "TrackletEdge",
    "TrackletGraphConfig",
    "TrackletPath",
    "build_tracklet_adjacency",
    "constant_velocity_edge_cost",
    "diverse_k_best_tracklet_paths",
    "k_best_tracklet_paths",
    "materialize_tracklet_path",
    "path_jaccard",
    "sort_tracklets",
    "tracklet_paths_to_dicts",
]
