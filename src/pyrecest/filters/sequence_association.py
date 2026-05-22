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
        Candidate identifier inside the frame. ``None`` denotes a missed
        detection and requires ``is_missed_detection=True``.
    unary_cost : float, optional
        Candidate-local cost before transition costs are added.
    is_missed_detection : bool, optional
        Whether this node is an explicit missed-detection branch.
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
        if self.candidate_index is None and not self.is_missed_detection:
            raise ValueError(
                "candidate_index=None is reserved for missed-detection nodes"
            )
        _validate_cost(self.unary_cost, "unary_cost")
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
        """Create an explicit missed-detection node for ``frame_index``."""
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
        """Return frame indices where the path selected a missed detection."""
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

    The dynamic program keeps the best predecessor for each node, then returns
    paths ending at the ``top_k_terminal_paths`` lowest-cost final-frame nodes.
    This mirrors the tracklet-Viterbi pattern used by RaFT-UAV and is not a full
    Yen-style k-shortest-path enumeration.
    """
    normalized_frames = _validate_frames(frames)
    top_k_terminal_paths = int(top_k_terminal_paths)
    if top_k_terminal_paths < 1:
        raise ValueError("top_k_terminal_paths must be positive")

    costs: list[np.ndarray] = [
        np.array([float(node.unary_cost) for node in normalized_frames[0]])
    ]
    miss_streaks: list[np.ndarray] = [
        np.array(
            [1 if node.is_missed_detection else 0 for node in normalized_frames[0]],
            dtype=int,
        )
    ]
    parents: list[np.ndarray] = [np.full(len(normalized_frames[0]), -1, dtype=int)]
    chosen_transition_costs: list[np.ndarray] = [
        np.zeros(len(normalized_frames[0]), dtype=float)
    ]

    for frame_pos in range(1, len(normalized_frames)):
        previous_frame = normalized_frames[frame_pos - 1]
        current_frame = normalized_frames[frame_pos]
        current_costs = np.empty(len(current_frame), dtype=float)
        current_miss_streaks = np.empty(len(current_frame), dtype=int)
        current_parents = np.empty(len(current_frame), dtype=int)
        current_transition_costs = np.empty(len(current_frame), dtype=float)

        for current_index, current_node in enumerate(current_frame):
            candidate_costs = np.empty(len(previous_frame), dtype=float)
            candidate_transition_costs = np.empty(len(previous_frame), dtype=float)
            for previous_index, previous_node in enumerate(previous_frame):
                context = SequenceTransitionContext(
                    frame_index=frame_pos,
                    previous_frame_index=previous_node.frame_index,
                    current_frame_index=current_node.frame_index,
                    previous_node_index=previous_index,
                    current_node_index=current_index,
                    previous_miss_streak=int(miss_streaks[-1][previous_index]),
                )
                transition_value = _validate_cost(
                    transition_cost(previous_node, current_node, context),
                    "transition_cost",
                )
                candidate_transition_costs[previous_index] = transition_value
                candidate_costs[previous_index] = (
                    costs[-1][previous_index] + transition_value
                )

            parent = int(np.argmin(candidate_costs))
            current_parents[current_index] = parent
            current_transition_costs[current_index] = candidate_transition_costs[parent]
            current_costs[current_index] = (
                float(current_node.unary_cost) + candidate_costs[parent]
            )
            current_miss_streaks[current_index] = (
                int(miss_streaks[-1][parent]) + 1
                if current_node.is_missed_detection
                else 0
            )

        costs.append(current_costs)
        miss_streaks.append(current_miss_streaks)
        parents.append(current_parents)
        chosen_transition_costs.append(current_transition_costs)

    terminal_count = min(top_k_terminal_paths, len(costs[-1]))
    terminal_indices = np.argsort(costs[-1])[:terminal_count]
    return tuple(
        _reconstruct_path(
            normalized_frames,
            parents,
            chosen_transition_costs,
            costs,
            int(terminal_index),
        )
        for terminal_index in terminal_indices
    )


def _validate_frames(
    frames: Sequence[Sequence[SequenceAssociationNode]],
) -> tuple[tuple[SequenceAssociationNode, ...], ...]:
    if not frames:
        raise ValueError("frames must contain at least one frame")
    normalized: list[tuple[SequenceAssociationNode, ...]] = []
    for frame_pos, frame in enumerate(frames):
        nodes = tuple(frame)
        if not nodes:
            raise ValueError(f"frame {frame_pos} must contain at least one node")
        for node in nodes:
            if not isinstance(node, SequenceAssociationNode):
                raise TypeError(
                    "all frames must contain SequenceAssociationNode objects"
                )
        normalized.append(nodes)
    return tuple(normalized)


def _reconstruct_path(
    frames: tuple[tuple[SequenceAssociationNode, ...], ...],
    parents: list[np.ndarray],
    chosen_transition_costs: list[np.ndarray],
    costs: list[np.ndarray],
    terminal_index: int,
) -> SequenceAssociationPath:
    node_index = int(terminal_index)
    path: list[SequenceAssociationNode] = []
    transition_values: list[float] = []
    for frame_pos in range(len(frames) - 1, -1, -1):
        path.append(frames[frame_pos][node_index])
        if frame_pos > 0:
            transition_values.append(
                float(chosen_transition_costs[frame_pos][node_index])
            )
        node_index = int(parents[frame_pos][node_index])
        if node_index < 0:
            break
    path.reverse()
    transition_values.reverse()
    return SequenceAssociationPath(
        total_cost=float(costs[-1][terminal_index]),
        nodes=tuple(path),
        transition_costs=tuple(transition_values),
    )


def _validate_cost(value: object, name: str) -> float:
    cost = float(value)
    if np.isnan(cost):
        raise ValueError(f"{name} must not be NaN")
    return cost
