"""Triplet-support consistency helpers for multi-session assignment costs.

The functions in this module operate only on pairwise cost matrices and session
indices.  They do not run an assignment solver.  Instead, they measure whether a
candidate edge is supported by a compatible two-edge path through a third
session and can add an optional bounded penalty to unsupported direct edges.

The adjustment is disabled by default.  Callers can therefore compute and audit
triplet support before deciding whether to use the adjusted matrices in a
multi-session assignment problem.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

SessionEdge = tuple[int, int]
SessionSizesInput = Mapping[int, int] | Sequence[int]


@dataclass(frozen=True)
class HigherOrderConsistencyConfig:
    """Configuration for bounded triplet-support penalties.

    For a direct edge with best third-session support cost ``s``, the unweighted
    penalty is ``clip(s - support_cost_cap, 0, max_penalty)``.  A missing finite
    support path receives ``max_penalty``.  ``triplet_weight`` scales the penalty
    before it is added to admissible direct-edge costs.

    A direct or supporting edge is considered forbidden when it is non-finite or
    greater than or equal to ``large_cost``.
    """

    triplet_weight: float = 0.0
    support_cost_cap: float = 4.0
    max_penalty: float = 2.0
    large_cost: float = 1.0e6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "triplet_weight",
            _normalize_finite_scalar(
                self.triplet_weight,
                name="triplet_weight",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "support_cost_cap",
            _normalize_finite_scalar(
                self.support_cost_cap,
                name="support_cost_cap",
            ),
        )
        object.__setattr__(
            self,
            "max_penalty",
            _normalize_finite_scalar(
                self.max_penalty,
                name="max_penalty",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "large_cost",
            _normalize_finite_scalar(
                self.large_cost,
                name="large_cost",
                minimum=0.0,
                strict_minimum=True,
            ),
        )
        if not math.isfinite(self.triplet_weight * self.max_penalty):
            raise ValueError("triplet_weight * max_penalty must be finite")

    @property
    def enabled(self) -> bool:
        """Return whether applying this configuration can change costs."""

        return self.triplet_weight > 0.0 and self.max_penalty > 0.0


def min_plus_triplet_support(
    left_costs: Any,
    right_costs: Any,
    *,
    large_cost: float = 1.0e6,
) -> np.ndarray:
    """Compute the exact min-plus product of two supporting cost matrices.

    ``left_costs`` must have shape ``(n_source, n_third)`` and ``right_costs``
    shape ``(n_third, n_target)``.  The returned entry ``(i, j)`` is

    ``min_k(left_costs[i, k] + right_costs[k, j])``.

    Non-finite entries and entries greater than or equal to ``large_cost`` are
    treated as forbidden.  An output entry is positive infinity when no finite
    two-edge support path exists.  The implementation iterates over the shared
    axis and only materializes sums between admissible entries, so sparse
    infinity-masked matrices do not require a dense three-dimensional tensor.
    """

    left = _as_cost_matrix(left_costs, name="left_costs")
    right = _as_cost_matrix(right_costs, name="right_costs")
    threshold = _normalize_finite_scalar(
        large_cost,
        name="large_cost",
        minimum=0.0,
        strict_minimum=True,
    )
    return _min_plus_triplet_support_normalized(
        left,
        right,
        large_cost=threshold,
    )


def triplet_support_costs(
    pairwise_costs: Mapping[SessionEdge, Any],
    *,
    edge: SessionEdge,
    session_sizes: SessionSizesInput | None = None,
    large_cost: float = 1.0e6,
) -> np.ndarray | None:
    """Return the best third-session support cost for one direct session edge.

    Backward, bridge, and forward contexts are considered:

    - ``third -> source`` together with ``third -> target``;
    - ``source -> third -> target``;
    - ``source -> third`` together with ``target -> third``.

    The minimum support cost over all available third-session contexts is
    returned.  ``None`` indicates that no complete third-session context exists;
    an infinity in the returned matrix indicates that contexts exist but no
    admissible two-edge path supports that candidate pair.
    """

    threshold = _normalize_finite_scalar(
        large_cost,
        name="large_cost",
        minimum=0.0,
        strict_minimum=True,
    )
    costs, sizes = _normalize_problem(pairwise_costs, session_sizes=session_sizes)
    normalized_edge = _normalize_edge(edge)
    if normalized_edge not in costs:
        raise KeyError(f"No pairwise cost matrix for edge {normalized_edge!r}")
    return _triplet_support_costs_normalized(
        costs,
        sizes,
        edge=normalized_edge,
        large_cost=threshold,
    )


def triplet_consistency_penalty(
    pairwise_costs: Mapping[SessionEdge, Any],
    *,
    edge: SessionEdge,
    session_sizes: SessionSizesInput | None = None,
    config: HigherOrderConsistencyConfig | Mapping[str, Any] | None = None,
) -> np.ndarray | None:
    """Return the unweighted bounded triplet-consistency penalty for one edge.

    ``None`` is returned when the edge has no available third-session context.
    """

    resolved = higher_order_consistency_config_from_mapping(config)
    costs, sizes = _normalize_problem(pairwise_costs, session_sizes=session_sizes)
    normalized_edge = _normalize_edge(edge)
    if normalized_edge not in costs:
        raise KeyError(f"No pairwise cost matrix for edge {normalized_edge!r}")
    support = _triplet_support_costs_normalized(
        costs,
        sizes,
        edge=normalized_edge,
        large_cost=resolved.large_cost,
    )
    if support is None:
        return None
    return _bounded_support_penalty(support, config=resolved)


def apply_higher_order_consistency(
    pairwise_costs: Mapping[SessionEdge, Any],
    *,
    session_sizes: SessionSizesInput | None = None,
    config: HigherOrderConsistencyConfig | Mapping[str, Any] | None = None,
) -> dict[SessionEdge, np.ndarray]:
    """Return copies of pairwise costs with optional triplet penalties added.

    Every input edge is preserved.  Forbidden direct edges and edges without a
    complete third-session context are copied unchanged.  The default
    configuration has zero weight and therefore performs no adjustment.
    """

    resolved = higher_order_consistency_config_from_mapping(config)
    costs, sizes = _normalize_problem(pairwise_costs, session_sizes=session_sizes)
    copied = {edge: matrix.copy() for edge, matrix in costs.items()}
    if not resolved.enabled:
        return copied

    adjusted: dict[SessionEdge, np.ndarray] = {}
    for edge, matrix in costs.items():
        support = _triplet_support_costs_normalized(
            costs,
            sizes,
            edge=edge,
            large_cost=resolved.large_cost,
        )
        if support is None:
            adjusted[edge] = matrix.copy()
            continue

        penalty = _bounded_support_penalty(support, config=resolved)
        admissible = _admissible_mask(matrix, large_cost=resolved.large_cost)
        edge_costs = matrix.copy()
        if np.any(admissible):
            with np.errstate(over="ignore", invalid="ignore"):
                penalized = (
                    matrix[admissible]
                    + resolved.triplet_weight * penalty[admissible]
                )
            penalized = np.nan_to_num(
                penalized,
                nan=resolved.large_cost,
                posinf=resolved.large_cost,
                neginf=-np.finfo(float).max,
            )
            edge_costs[admissible] = np.minimum(penalized, resolved.large_cost)
        adjusted[edge] = edge_costs
    return adjusted


def higher_order_consistency_config_from_mapping(
    config: HigherOrderConsistencyConfig | Mapping[str, Any] | None,
) -> HigherOrderConsistencyConfig:
    """Normalize an optional higher-order-consistency configuration."""

    if config is None:
        return HigherOrderConsistencyConfig()
    if isinstance(config, HigherOrderConsistencyConfig):
        return config
    if not isinstance(config, Mapping):
        raise ValueError(
            "config must be a HigherOrderConsistencyConfig, mapping, or None"
        )
    return HigherOrderConsistencyConfig(**dict(config))


def _triplet_support_costs_normalized(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    session_sizes: Mapping[int, int],
    *,
    edge: SessionEdge,
    large_cost: float,
) -> np.ndarray | None:
    contexts = _triplet_support_contexts(
        pairwise_costs,
        session_indices=tuple(sorted(session_sizes)),
        edge=edge,
    )
    if not contexts:
        return None

    support = np.full(pairwise_costs[edge].shape, np.inf, dtype=float)
    for left, right in contexts:
        context_support = _min_plus_triplet_support_normalized(
            left,
            right,
            large_cost=large_cost,
        )
        support = np.minimum(support, context_support)
    return support


def _triplet_support_contexts(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    session_indices: Sequence[int],
    edge: SessionEdge,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    source, target = edge
    contexts: list[tuple[np.ndarray, np.ndarray]] = []

    for third in session_indices:
        if third < source:
            third_to_source = pairwise_costs.get((third, source))
            third_to_target = pairwise_costs.get((third, target))
            if third_to_source is not None and third_to_target is not None:
                contexts.append((third_to_source.T, third_to_target))
        elif source < third < target:
            source_to_third = pairwise_costs.get((source, third))
            third_to_target = pairwise_costs.get((third, target))
            if source_to_third is not None and third_to_target is not None:
                contexts.append((source_to_third, third_to_target))
        elif third > target:
            source_to_third = pairwise_costs.get((source, third))
            target_to_third = pairwise_costs.get((target, third))
            if source_to_third is not None and target_to_third is not None:
                contexts.append((source_to_third, target_to_third.T))

    return tuple(contexts)


def _min_plus_triplet_support_normalized(
    left: np.ndarray,
    right: np.ndarray,
    *,
    large_cost: float,
) -> np.ndarray:
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            "left_costs columns must match right_costs rows for min-plus support"
        )

    support = np.full((left.shape[0], right.shape[1]), np.inf, dtype=float)
    for shared_index in range(left.shape[1]):
        left_values = left[:, shared_index]
        right_values = right[shared_index, :]
        left_indices = np.flatnonzero(
            _admissible_mask(left_values, large_cost=large_cost)
        )
        right_indices = np.flatnonzero(
            _admissible_mask(right_values, large_cost=large_cost)
        )
        if left_indices.size == 0 or right_indices.size == 0:
            continue

        with np.errstate(over="ignore", invalid="ignore"):
            candidate = (
                left_values[left_indices, None] + right_values[None, right_indices]
            )
        candidate = np.nan_to_num(
            candidate,
            nan=np.inf,
            posinf=np.inf,
            neginf=-np.finfo(float).max,
        )
        index = np.ix_(left_indices, right_indices)
        support[index] = np.minimum(support[index], candidate)
    return support


def _bounded_support_penalty(
    support: np.ndarray,
    *,
    config: HigherOrderConsistencyConfig,
) -> np.ndarray:
    penalty = np.full(support.shape, config.max_penalty, dtype=float)
    finite = np.isfinite(support)
    if np.any(finite):
        with np.errstate(over="ignore", invalid="ignore"):
            excess = support[finite] - config.support_cost_cap
        excess = np.nan_to_num(
            excess,
            nan=config.max_penalty,
            posinf=config.max_penalty,
            neginf=0.0,
        )
        penalty[finite] = np.clip(excess, 0.0, config.max_penalty)
    return penalty


def _normalize_problem(
    pairwise_costs: Mapping[SessionEdge, Any],
    *,
    session_sizes: SessionSizesInput | None,
) -> tuple[dict[SessionEdge, np.ndarray], dict[int, int]]:
    if not isinstance(pairwise_costs, Mapping):
        raise ValueError(
            "pairwise_costs must be a mapping from session edges to matrices"
        )

    normalized_costs: dict[SessionEdge, np.ndarray] = {}
    for raw_edge, raw_matrix in pairwise_costs.items():
        edge = _normalize_edge(raw_edge)
        if edge in normalized_costs:
            raise ValueError(f"Duplicate pairwise cost edge {edge!r}")
        normalized_costs[edge] = _as_cost_matrix(
            raw_matrix,
            name=f"pairwise_costs[{edge!r}]",
        )

    normalized_sizes = _normalize_session_sizes(session_sizes)
    for (source, target), matrix in normalized_costs.items():
        _record_session_size(normalized_sizes, source, matrix.shape[0])
        _record_session_size(normalized_sizes, target, matrix.shape[1])

    return normalized_costs, normalized_sizes


def _normalize_session_sizes(
    session_sizes: SessionSizesInput | None,
) -> dict[int, int]:
    if session_sizes is None:
        return {}
    if isinstance(session_sizes, Mapping):
        items = session_sizes.items()
    else:
        if isinstance(session_sizes, (str, bytes, bytearray)):
            raise ValueError("session_sizes must be a mapping, sequence, or None")
        try:
            items = enumerate(session_sizes)
        except TypeError as exc:
            raise ValueError(
                "session_sizes must be a mapping, sequence, or None"
            ) from exc

    normalized: dict[int, int] = {}
    for raw_session, raw_size in items:
        session = _normalize_nonnegative_integer(raw_session, name="session index")
        size = _normalize_nonnegative_integer(
            raw_size,
            name=f"session_sizes[{session}]",
        )
        if session in normalized:
            raise ValueError(f"Duplicate session size for session {session}")
        normalized[session] = size
    return normalized


def _record_session_size(
    session_sizes: dict[int, int],
    session: int,
    observed_size: int,
) -> None:
    expected = session_sizes.get(session)
    if expected is None:
        session_sizes[session] = int(observed_size)
    elif expected != int(observed_size):
        raise ValueError(
            f"Session {session} has size {observed_size} in a pairwise matrix, "
            f"but session_sizes specifies {expected}"
        )


def _normalize_edge(edge: Any) -> SessionEdge:
    if isinstance(edge, (str, bytes, bytearray)):
        raise ValueError("Session edges must contain exactly two indices")
    try:
        values = tuple(edge)
    except TypeError as exc:
        raise ValueError("Session edges must contain exactly two indices") from exc
    if len(values) != 2:
        raise ValueError("Session edges must contain exactly two indices")

    source = _normalize_nonnegative_integer(values[0], name="source session")
    target = _normalize_nonnegative_integer(values[1], name="target session")
    if source >= target:
        raise ValueError("Session edges must satisfy source < target")
    return source, target


def _as_cost_matrix(value: Any, *, name: str) -> np.ndarray:
    if np.ma.is_masked(value):
        raise ValueError(f"{name} must be a real-valued numeric matrix")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a real-valued numeric matrix") from exc

    if raw.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if raw.dtype.kind in {"b", "c", "M", "m", "S", "U"}:
        raise ValueError(f"{name} must be a real-valued numeric matrix")
    if raw.dtype.kind == "O":
        for item in raw.flat:
            if (
                item is None
                or np.ma.is_masked(item)
                or isinstance(item, (bool, np.bool_, complex, np.complexfloating))
                or not isinstance(item, Real)
            ):
                raise ValueError(f"{name} must be a real-valued numeric matrix")

    try:
        matrix = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real-valued numeric matrix") from exc
    if np.any(np.isnan(matrix)) or np.any(np.isneginf(matrix)):
        raise ValueError(
            f"{name} may only contain finite values or positive infinity"
        )
    return matrix


def _admissible_mask(values: np.ndarray, *, large_cost: float) -> np.ndarray:
    return np.isfinite(values) & (values < large_cost)


def _normalize_nonnegative_integer(value: Any, *, name: str) -> int:
    message = f"{name} must be a non-negative integer"
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(message)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(message) from exc
    if raw.shape != () or raw.dtype.kind in {"b", "c", "M", "m", "S", "U"}:
        raise ValueError(message)

    scalar = raw.item()
    if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, Real):
        raise ValueError(message)
    if isinstance(scalar, Integral):
        parsed = int(scalar)
    else:
        try:
            as_float = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if not math.isfinite(as_float) or not as_float.is_integer():
            raise ValueError(message)
        parsed = int(as_float)
    if parsed < 0:
        raise ValueError(message)
    return parsed


def _normalize_finite_scalar(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    message = f"{name} must be a finite scalar"
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(message)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(message) from exc
    if raw.shape != () or raw.dtype.kind in {"b", "c", "M", "m", "S", "U"}:
        raise ValueError(message)

    scalar = raw.item()
    if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, Real):
        raise ValueError(message)
    try:
        parsed = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(parsed):
        raise ValueError(message)
    if minimum is not None:
        invalid = parsed <= minimum if strict_minimum else parsed < minimum
        if invalid:
            comparator = "greater than" if strict_minimum else "at least"
            raise ValueError(f"{name} must be {comparator} {minimum}")
    return parsed


__all__ = (
    "HigherOrderConsistencyConfig",
    "SessionEdge",
    "apply_higher_order_consistency",
    "higher_order_consistency_config_from_mapping",
    "min_plus_triplet_support",
    "triplet_consistency_penalty",
    "triplet_support_costs",
)
