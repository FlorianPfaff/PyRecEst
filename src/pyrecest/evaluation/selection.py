"""Deterministic score-based selection helpers.

The functions in this module are intentionally domain-neutral. They are useful
for experiments that need reproducible top-k selection, quantile-tail masks, or
retention-constrained selection that protects a low-reliability tail. Examples
include measurement-reliability ablations, particle/track pruning diagnostics,
and point-set or extended-object evaluation pipelines.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

TailSide = Literal["lower", "upper"]


def _normalize_nonnegative_integer(value, name: str) -> int:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be a non-negative integer.")

    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    if isinstance(scalar, (int, np.integer)):
        integer = int(scalar)
    else:
        try:
            scalar_float = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a non-negative integer.") from exc
        if not np.isfinite(scalar_float) or not scalar_float.is_integer():
            raise ValueError(f"{name} must be a non-negative integer.")
        integer = int(scalar_float)

    if integer < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return integer


def _normalize_finite_scalar(value, message: str) -> float:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(message)

    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    try:
        result = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(result):
        raise ValueError(message)
    return result


def _normalize_bounded_fraction(
    value,
    *,
    lower: float,
    upper: float,
    include_lower: bool,
    include_upper: bool,
    message: str,
) -> float:
    fraction = _normalize_finite_scalar(value, message)
    lower_ok = fraction >= lower if include_lower else fraction > lower
    upper_ok = fraction <= upper if include_upper else fraction < upper
    if not lower_ok or not upper_ok:
        raise ValueError(message)
    return fraction


def sanitized_score_vector(values, *, nonnegative: bool = True) -> np.ndarray:
    """Return a finite one-dimensional ``float64`` score vector.

    Non-finite entries are mapped to zero. By default negative values are also
    clipped to zero, matching the common interpretation of scores as confidence,
    reliability, probability, or non-negative utility.
    """
    clean = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    clean[~np.isfinite(clean)] = 0.0
    if nonnegative:
        clean = np.maximum(clean, 0.0)
    return clean


def retained_count_from_fraction(
    item_count: int,
    retention_fraction: float,
    *,
    min_count: int = 1,
) -> int:
    """Convert a retention fraction to a deterministic retained count.

    ``min_count`` is applied only when ``item_count > 0`` and
    ``retention_fraction > 0``. Set ``min_count=0`` for exact zero-retention
    behavior.
    """
    count = _normalize_nonnegative_integer(item_count, "item_count")
    fraction = _normalize_bounded_fraction(
        retention_fraction,
        lower=0.0,
        upper=1.0,
        include_lower=True,
        include_upper=True,
        message="retention_fraction must be finite and in [0, 1].",
    )
    minimum = _normalize_nonnegative_integer(min_count, "min_count")
    if count == 0 or fraction == 0.0:
        return 0
    return int(min(count, max(minimum, np.ceil(fraction * count))))


def top_count_mask(
    scores,
    retained_count: int,
    *,
    tie_break_scores=None,
    largest: bool = True,
    sanitize_nonnegative: bool = True,
) -> np.ndarray:
    """Return a deterministic mask selecting ``retained_count`` score entries.

    Ties are resolved by the optional ``tie_break_scores`` and then by increasing
    original index, making the selection reproducible across NumPy versions.
    """
    primary = sanitized_score_vector(scores, nonnegative=sanitize_nonnegative)
    count = int(primary.shape[0])
    retained = _normalize_nonnegative_integer(retained_count, "retained_count")
    if retained > count:
        raise ValueError("retained_count must be in [0, len(scores)].")
    mask = np.zeros((count,), dtype=bool)
    if retained == 0:
        return mask

    if tie_break_scores is None:
        secondary = None
    else:
        secondary = sanitized_score_vector(
            tie_break_scores,
            nonnegative=sanitize_nonnegative,
        )
        if secondary.shape != primary.shape:
            raise ValueError("tie_break_scores must have the same length as scores.")

    indices = np.arange(count, dtype=np.int64)
    if largest:
        if secondary is None:
            order = np.lexsort((indices, -primary))
        else:
            order = np.lexsort((indices, -secondary, -primary))
    elif secondary is None:
        order = np.lexsort((indices, primary))
    else:
        order = np.lexsort((indices, secondary, primary))
    mask[order[:retained]] = True
    return mask


def top_fraction_mask(
    scores,
    retention_fraction: float,
    *,
    tie_break_scores=None,
    largest: bool = True,
    min_count: int = 1,
    sanitize_nonnegative: bool = True,
) -> np.ndarray:
    """Return a top-k mask whose size is derived from a retention fraction."""
    primary = sanitized_score_vector(scores, nonnegative=sanitize_nonnegative)
    retained = retained_count_from_fraction(
        int(primary.shape[0]),
        retention_fraction,
        min_count=min_count,
    )
    return top_count_mask(
        primary,
        retained,
        tie_break_scores=tie_break_scores,
        largest=largest,
        sanitize_nonnegative=sanitize_nonnegative,
    )


def quantile_tail_threshold(
    reliability_scores,
    quantile: float,
    *,
    tail: TailSide = "lower",
    sanitize_nonnegative: bool = True,
) -> float:
    """Return the threshold separating a reliability-score quantile tail."""
    scores = sanitized_score_vector(
        reliability_scores,
        nonnegative=sanitize_nonnegative,
    )
    q = float(quantile)
    if not np.isfinite(q) or not 0.0 < q < 1.0:
        raise ValueError("quantile must be finite and in (0, 1).")
    if tail not in {"lower", "upper"}:
        raise ValueError("tail must be 'lower' or 'upper'.")
    if scores.size == 0:
        return float("nan")
    return float(np.quantile(scores, q if tail == "lower" else 1.0 - q))


def quantile_tail_mask(
    reliability_scores,
    quantile: float,
    *,
    tail: TailSide = "lower",
    sanitize_nonnegative: bool = True,
) -> np.ndarray:
    """Return a boolean mask for the lower or upper reliability-score tail."""
    scores = sanitized_score_vector(
        reliability_scores,
        nonnegative=sanitize_nonnegative,
    )
    if scores.size == 0:
        return np.zeros((0,), dtype=bool)
    threshold = quantile_tail_threshold(
        scores,
        quantile,
        tail=tail,
        sanitize_nonnegative=sanitize_nonnegative,
    )
    if tail == "lower":
        return scores <= threshold
    return scores >= threshold


def protected_tail_topk_mask(
    primary_scores,
    tail_scores,
    reliability_scores,
    retention_fraction: float,
    *,
    tail_quantile: float,
    tail: TailSide = "lower",
    min_count: int = 1,
    sanitize_nonnegative: bool = True,
) -> np.ndarray:
    """Select a fixed-size subset while preserving proportional tail capacity.

    The candidate set is split at ``tail_quantile`` of ``reliability_scores``.
    The tail receives the same retention fraction as the full set, ranked by
    ``tail_scores``. The complement receives the remaining retained budget,
    ranked by ``primary_scores``. If either side is empty the function falls
    back to ordinary top-fraction selection by ``primary_scores``.
    """
    primary = sanitized_score_vector(primary_scores, nonnegative=sanitize_nonnegative)
    tail_rank = sanitized_score_vector(tail_scores, nonnegative=sanitize_nonnegative)
    reliability = sanitized_score_vector(
        reliability_scores,
        nonnegative=sanitize_nonnegative,
    )
    if primary.shape != tail_rank.shape or primary.shape != reliability.shape:
        raise ValueError(
            "primary_scores, tail_scores, and reliability_scores must have the same length."
        )
    count = int(primary.shape[0])
    if count == 0:
        return np.zeros((0,), dtype=bool)

    retained_total = retained_count_from_fraction(
        count,
        retention_fraction,
        min_count=min_count,
    )
    tail_mask = quantile_tail_mask(
        reliability,
        tail_quantile,
        tail=tail,
        sanitize_nonnegative=sanitize_nonnegative,
    )
    tail_indices = np.flatnonzero(tail_mask)
    complement_indices = np.flatnonzero(~tail_mask)
    if tail_indices.size == 0 or complement_indices.size == 0:
        return top_count_mask(
            primary,
            retained_total,
            sanitize_nonnegative=sanitize_nonnegative,
        )

    tail_quota = min(
        int(tail_indices.size),
        retained_count_from_fraction(
            int(tail_indices.size),
            retention_fraction,
            min_count=min_count,
        ),
    )
    complement_quota = retained_total - tail_quota
    if complement_quota > int(complement_indices.size):
        tail_quota = min(
            int(tail_indices.size),
            tail_quota + complement_quota - int(complement_indices.size),
        )
        complement_quota = int(complement_indices.size)
    if complement_quota < 0:
        tail_quota = max(0, tail_quota + complement_quota)
        complement_quota = 0

    mask = np.zeros((count,), dtype=bool)
    if tail_quota > 0:
        local_tail = top_count_mask(
            tail_rank[tail_indices],
            tail_quota,
            sanitize_nonnegative=sanitize_nonnegative,
        )
        mask[tail_indices[local_tail]] = True
    if complement_quota > 0:
        local_complement = top_count_mask(
            primary[complement_indices],
            complement_quota,
            sanitize_nonnegative=sanitize_nonnegative,
        )
        mask[complement_indices[local_complement]] = True

    missing = retained_total - int(mask.sum())
    if missing > 0:
        remaining = np.flatnonzero(~mask)
        fill = top_count_mask(
            primary[remaining],
            min(int(missing), int(remaining.size)),
            sanitize_nonnegative=sanitize_nonnegative,
        )
        mask[remaining[fill]] = True
    return mask


def tail_rescue_quota_count(retained_count: int, *, rescue_fraction: float) -> int:
    """Return a bounded tail-rescue quota inside a retained budget."""
    retained = _normalize_nonnegative_integer(retained_count, "retained_count")
    rescue = _normalize_bounded_fraction(
        rescue_fraction,
        lower=0.0,
        upper=1.0,
        include_lower=False,
        include_upper=True,
        message="rescue_fraction must be finite and in (0, 1].",
    )
    if retained == 0:
        return 0
    return int(min(retained, max(1, np.ceil(rescue * retained))))


def tail_rescue_topk_mask(
    primary_scores,
    tail_scores,
    reliability_scores,
    retention_fraction: float,
    *,
    tail_quantile: float,
    rescue_fraction: float,
    tail: TailSide = "lower",
    min_count: int = 1,
    sanitize_nonnegative: bool = True,
) -> np.ndarray:
    """Top-k selection with a bounded quota rescued from a reliability tail."""
    primary = sanitized_score_vector(primary_scores, nonnegative=sanitize_nonnegative)
    tail_rank = sanitized_score_vector(tail_scores, nonnegative=sanitize_nonnegative)
    reliability = sanitized_score_vector(
        reliability_scores,
        nonnegative=sanitize_nonnegative,
    )
    if primary.shape != tail_rank.shape or primary.shape != reliability.shape:
        raise ValueError(
            "primary_scores, tail_scores, and reliability_scores must have the same length."
        )
    count = int(primary.shape[0])
    if count == 0:
        return np.zeros((0,), dtype=bool)

    retained_total = retained_count_from_fraction(
        count,
        retention_fraction,
        min_count=min_count,
    )
    tail_mask = quantile_tail_mask(
        reliability,
        tail_quantile,
        tail=tail,
        sanitize_nonnegative=sanitize_nonnegative,
    )
    tail_indices = np.flatnonzero(tail_mask)
    if tail_indices.size == 0:
        return top_count_mask(
            primary,
            retained_total,
            sanitize_nonnegative=sanitize_nonnegative,
        )

    mask = top_count_mask(
        primary,
        retained_total,
        sanitize_nonnegative=sanitize_nonnegative,
    )
    rescue_quota = min(
        int(tail_indices.size),
        tail_rescue_quota_count(
            retained_total,
            rescue_fraction=rescue_fraction,
        ),
    )
    current_tail = int(np.sum(mask[tail_indices]))
    missing_tail = rescue_quota - current_tail
    if missing_tail <= 0:
        return mask

    tail_candidates = tail_indices[~mask[tail_indices]]
    if tail_candidates.size == 0:
        return mask
    add_mask = top_count_mask(
        tail_rank[tail_candidates],
        min(missing_tail, int(tail_candidates.size)),
        sanitize_nonnegative=sanitize_nonnegative,
    )
    add_indices = tail_candidates[add_mask]
    selected_non_tail = np.flatnonzero(mask & ~tail_mask)
    if add_indices.size > selected_non_tail.size:
        add_indices = add_indices[: selected_non_tail.size]
    if add_indices.size == 0:
        return mask

    drop_order = np.lexsort((selected_non_tail, primary[selected_non_tail]))
    drop_indices = selected_non_tail[drop_order[: add_indices.size]]
    mask[drop_indices] = False
    mask[add_indices] = True
    return mask
