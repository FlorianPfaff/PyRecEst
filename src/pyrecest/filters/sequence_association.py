# pylint: disable=too-many-locals
"""Sequence-level association helpers based on Viterbi dynamic programming.

The helpers in this module contain the dataset-neutral core of a tracklet-level
association strategy: represent each scan/frame as a list of candidate nodes,
provide a transition-cost function, and recover the lowest-cost coherent path.
Downstream projects can keep domain-specific candidate scoring outside PyRecEst
while reusing the dynamic-programming machinery.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SequenceAssociationNode:
    """One candidate in one sequence-association frame.

    Parameters
    ----------
    frame_index : int
        Original frame or event index.  The value does not need to be dense;
        it is preserved in returned paths for downstream bookkeeping.
    candidate_index : int or None
        Candidate identifier inside the frame. ``None`` denotes an explicit
        gap and requires ``is_missed_detection=True``.
    unary_cost : float, optional
        Candidate-local cost before transition costs are added.
    is_missed_detection : bool, optional
        Whether this node is an explicit gap branch.
    payload : object, optional
        Optional domain object, row, measurement, or label carried through the
        solver unchanged.
    metadata : dict, optional
        Optional immutable-by-convention diagnostics carried through unchanged.
    """

    frame_index: int
    candidate_index: int | None
    unary_cost: float = 0.0
    is_missed_detection: bool = False
    payload: Any | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        frame_index = _validate_integer(self.frame_index, "frame_index")
        is_missed_detection = _validate_bool_flag(
            self.is_missed_detection,
            "is_missed_detection",
        )
        if self.candidate_index is None:
            if not is_missed_detection:
                raise ValueError(
                    "candidate_index=None is reserved for explicit gap nodes"
                )
            candidate_index = None
        else:
            if is_missed_detection:
                raise ValueError("candidate_index must be None for explicit gap nodes")
            candidate_index = _validate_integer(self.candidate_index, "candidate_index")
        unary_cost = _validate_cost(self.unary_cost, "unary_cost")
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "candidate_index", candidate_index)
        object.__setattr__(self, "unary_cost", unary_cost)
        object.__setattr__(self, "is_missed_detection", is_missed_detection)
        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def missed_detection(
        cls,
        frame_index: int,
        *,
        unary_cost: float = 0.0,
        payload: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SequenceAssociationNode":
        """Create an explicit gap node for ``frame_index``."""
        return cls(
            frame_index=frame_index,
            candidate_index=None,
            unary_cost=unary_cost,
            is_missed_detection=True,
            payload=payload,
            metadata=metadata,
        )


@dataclass(frozen=True)
class SequenceTransitionContext:
    """Context passed to a sequence-association transition-cost function."""

    frame_index: int
    previous_frame_index: int
    current_frame_index: int
    previous_node_index: int
    current_node_index: int
    previous_miss_streak: int


@dataclass(frozen=True)
class SequenceAssociationPath:
    """Viterbi path returned by sequence association."""

    total_cost: float
    nodes: tuple[SequenceAssociationNode, ...]
    transition_costs: tuple[float, ...]

    @property
    def candidate_indices(self) -> tuple[int | None, ...]:
        """Return the candidate index chosen in each frame."""
        return tuple(node.candidate_index for node in self.nodes)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        """Return original frame indices represented by this path."""
        return tuple(node.frame_index for node in self.nodes)

    @property
    def missed_detection_frame_indices(self) -> tuple[int, ...]:
        """Return frame indices where the path selected an explicit gap."""
        return tuple(
            node.frame_index for node in self.nodes if node.is_missed_detection
        )

    @property
    def payloads(self) -> tuple[Any | None, ...]:
        """Return node payloads in path order."""
        return tuple(node.payload for node in self.nodes)


TransitionCostFn = Callable[
    [SequenceAssociationNode, SequenceAssociationNode, SequenceTransitionContext],
    float,
]


def solve_viterbi_sequence_association(
    frames: Sequence[Sequence[SequenceAssociationNode]],
    transition_cost: TransitionCostFn,
) -> SequenceAssociationPath:
    """Return the lowest-cost path through candidate frames.

    ``frames`` is a sequence of scans/time steps, each containing one or more
    :class:`SequenceAssociationNode` objects.  ``transition_cost`` scores a
    transition from a node in frame ``k-1`` to a node in frame ``k``.  The total
    path cost is the sum of all selected unary costs and transition costs.
    """
    return solve_top_k_viterbi_sequence_associations(
        frames,
        transition_cost,
        top_k_terminal_paths=1,
    )[0]


def solve_top_k_viterbi_sequence_associations(
    frames: Sequence[Sequence[SequenceAssociationNode]],
    transition_cost: TransitionCostFn,
    *,
    top_k_terminal_paths: int = 1,
) -> tuple[SequenceAssociationPath, ...]:
    """Return best Viterbi paths for the lowest-cost terminal nodes.

    The dynamic program keeps the best predecessor for each
    ``(node, missed-detection streak)`` state, then returns paths ending at the
    ``top_k_terminal_paths`` lowest-cost final-frame nodes.  Tracking the streak
    as part of the state is necessary because transition costs can depend on
    ``SequenceTransitionContext.previous_miss_streak``.  This mirrors the
    tracklet-Viterbi pattern used by RaFT-UAV and is not a full Yen-style
    k-shortest-path enumeration.
    """
    normalized_frames = _validate_frames(frames)
    top_k_terminal_paths = _validate_positive_integer(
        top_k_terminal_paths,
        "top_k_terminal_paths",
    )

    initial_streaks = [
        1 if node.is_missed_detection else 0 for node in normalized_frames[0]
    ]
    costs: list[list[dict[int, float]]] = [
        [
            {miss_streak: float(node.unary_cost)}
            for node, miss_streak in zip(normalized_frames[0], initial_streaks)
        ]
    ]
    parents: list[list[dict[int, tuple[int, int]]]] = [
        [{miss_streak: (-1, -1)} for miss_streak in initial_streaks]
    ]
    chosen_transition_costs: list[list[dict[int, float]]] = [
        [{miss_streak: 0.0} for miss_streak in initial_streaks]
    ]

    for frame_pos in range(1, len(normalized_frames)):
        previous_frame = normalized_frames[frame_pos - 1]
        current_frame = normalized_frames[frame_pos]
        current_costs: list[dict[int, float]] = [{} for _current_node in current_frame]
        current_parents: list[dict[int, tuple[int, int]]] = [
            {} for _current_node in current_frame
        ]
        current_transition_costs: list[dict[int, float]] = [
            {} for _current_node in current_frame
        ]

        for current_index, current_node in enumerate(current_frame):
            for previous_index, previous_node in enumerate(previous_frame):
                for previous_miss_streak, previous_cost in costs[-1][
                    previous_index
                ].items():
                    context = SequenceTransitionContext(
                        frame_index=frame_pos,
                        previous_frame_index=previous_node.frame_index,
                        current_frame_index=current_node.frame_index,
                        previous_node_index=previous_index,
                        current_node_index=current_index,
                        previous_miss_streak=previous_miss_streak,
                    )
                    transition_value = _validate_cost(
                        transition_cost(previous_node, current_node, context),
                        "transition_cost",
                    )
                    current_miss_streak = (
                        previous_miss_streak + 1
                        if current_node.is_missed_detection
                        else 0
                    )
                    candidate_cost = (
                        previous_cost
                        + transition_value
                        + float(current_node.unary_cost)
                    )
                    best_cost = current_costs[current_index].get(current_miss_streak)
                    if best_cost is None or candidate_cost < best_cost:
                        current_costs[current_index][
                            current_miss_streak
                        ] = candidate_cost
                        current_parents[current_index][current_miss_streak] = (
                            previous_index,
                            previous_miss_streak,
                        )
                        current_transition_costs[current_index][
                            current_miss_streak
                        ] = transition_value

        costs.append(current_costs)
        parents.append(current_parents)
        chosen_transition_costs.append(current_transition_costs)

    terminal_hypotheses: list[tuple[float, int, int]] = []
    for terminal_index, terminal_costs in enumerate(costs[-1]):
        terminal_miss_streak = min(terminal_costs, key=terminal_costs.__getitem__)
        terminal_hypotheses.append(
            (
                terminal_costs[terminal_miss_streak],
                terminal_index,
                terminal_miss_streak,
            )
        )
    terminal_hypotheses.sort(key=lambda hypothesis: (hypothesis[0], hypothesis[1]))

    terminal_count = min(top_k_terminal_paths, len(terminal_hypotheses))
    return tuple(
        _reconstruct_path(
            normalized_frames,
            parents,
            chosen_transition_costs,
            costs,
            terminal_index,
            terminal_miss_streak,
        )
        for _, terminal_index, terminal_miss_streak in terminal_hypotheses[
            :terminal_count
        ]
    )


def _validate_frames(
    frames: Sequence[Sequence[SequenceAssociationNode]],
) -> tuple[tuple[SequenceAssociationNode, ...], ...]:
    normalized: list[tuple[SequenceAssociationNode, ...]] = []
    try:
        frame_iterator = enumerate(frames)
    except TypeError as exc:
        raise ValueError("frames must contain at least one frame") from exc

    for frame_pos, frame in frame_iterator:
        nodes = tuple(frame)
        if not nodes:
            raise ValueError(f"frame {frame_pos} must contain at least one node")
        for node in nodes:
            if not isinstance(node, SequenceAssociationNode):
                raise TypeError(
                    "all frames must contain SequenceAssociationNode objects"
                )
        normalized.append(nodes)
    if not normalized:
        raise ValueError("frames must contain at least one frame")
    return tuple(normalized)


def _reconstruct_path(
    frames: tuple[tuple[SequenceAssociationNode, ...], ...],
    parents: list[list[dict[int, tuple[int, int]]]],
    chosen_transition_costs: list[list[dict[int, float]]],
    costs: list[list[dict[int, float]]],
    terminal_index: int,
    terminal_miss_streak: int,
) -> SequenceAssociationPath:
    total_cost = float(costs[-1][terminal_index][terminal_miss_streak])
    node_index = int(terminal_index)
    miss_streak = int(terminal_miss_streak)
    path: list[SequenceAssociationNode] = []
    transition_values: list[float] = []
    for frame_pos in range(len(frames) - 1, -1, -1):
        path.append(frames[frame_pos][node_index])
        if frame_pos > 0:
            transition_values.append(
                float(chosen_transition_costs[frame_pos][node_index][miss_streak])
            )
            node_index, miss_streak = parents[frame_pos][node_index][miss_streak]
    path.reverse()
    transition_values.reverse()
    return SequenceAssociationPath(
        total_cost=total_cost,
        nodes=tuple(path),
        transition_costs=tuple(transition_values),
    )


def _validate_bool_flag(value: object, name: str) -> bool:
    message = f"{name} must be a bool"
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if value_array.ndim != 0 or value_array.dtype != np.bool_:
        raise ValueError(message)
    return bool(value_array.item())


def _validate_integer(value: object, name: str) -> int:
    message = f"{name} must be an integer"
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if value_array.ndim != 0 or value_array.dtype == np.bool_:
        raise ValueError(message)
    if value_array.dtype.kind in {"S", "U", "c"}:
        raise ValueError(message)

    scalar = value_array.item()
    if isinstance(
        scalar,
        (bool, np.bool_, str, bytes, bytearray, complex, np.complexfloating),
    ):
        raise ValueError(message)
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    if (
        isinstance(scalar, (float, np.floating))
        and np.isfinite(scalar)
        and float(scalar).is_integer()
    ):
        return int(scalar)
    raise ValueError(message)


def _validate_cost(value: object, name: str) -> float:
    numeric_message = f"{name} must be a scalar numeric cost"
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(numeric_message) from exc

    if value_array.ndim != 0 or value_array.dtype.kind in {"b", "S", "U", "c"}:
        raise ValueError(numeric_message)

    scalar = value_array.item()
    if isinstance(
        scalar,
        (bool, np.bool_, str, bytes, bytearray, complex, np.complexfloating),
    ):
        raise ValueError(numeric_message)

    try:
        cost = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(numeric_message) from exc
    if not np.isfinite(cost):
        raise ValueError(f"{name} must be finite")
    return cost


def _validate_positive_integer(value: object, name: str) -> int:
    message = f"{name} must be a positive integer"
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if value_array.ndim != 0 or value_array.dtype == np.bool_:
        raise ValueError(message)

    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(scalar, (int, np.integer)):
        result = int(scalar)
    elif (
        isinstance(scalar, (float, np.floating))
        and np.isfinite(scalar)
        and float(scalar).is_integer()
    ):
        result = int(scalar)
    else:
        raise ValueError(message)

    if result < 1:
        raise ValueError(message)
    return result
